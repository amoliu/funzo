"""
Bayesian inverse reinforcement learning

"""

from __future__ import division

import six
import numpy as np

from abc import abstractmethod, ABCMeta
from tqdm import tqdm
from copy import deepcopy
from six.moves import range, zip

from scipy.optimize import minimize
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal

from .base import IRLSolver
from ..base import Model
from ..utils.validation import check_random_state
from ..utils.data_structures import Trace


__all__ = ['BIRL']


class BIRL(IRLSolver):
    """ Bayesian IRL algorithm

    BIRL algorithm that seeks to find a reward function underlying a set of
    expert demonstrations by computing the posterior of the reward distribution
    :math:`p(r | \Xi)`.

    These algorithms typically summarize the distribution by taking a single
    value such as the mean.

    Parameters
    ----------
    planner : a callable
        A planner for MDP e.g. policy iteration as a callable
    prior : :class:`RewardPrior` or derivative object
        Reward prior callable object
    inference : str, optional (default='PW')
        Inference procedure ['PW', 'MPW', 'MAP']
    beta : float, optional (default=0.7)
        Expert optimality parameter for the reward likelihood term in the
        product of exponential distributions

    Attributes
    ----------
    _prior : :class:`RewardPrior` or derivative object
        Reference to the reward prior callable object
    _beta : float, optional (default=0.9)
        Expert optimality parameter for the reward likelihood term in the
        product of exponential distributions

    """

    def __init__(self, planner, prior, inference='PW', beta=0.7, delta=0.2,
                 burn_ratio=0.27, max_iter=100, random_state=None):
        super(BIRL, self).__init__(planner)
        self._prior = prior
        self._beta = beta
        self._inference = inference
        self._rng = check_random_state(random_state)

        if 0 >= max_iter > np.inf:
            raise ValueError('No. of iterations must be in (0, inf)')
        self._max_iter = max_iter

        if self._inference in ['PW', 'MPW']:
            if 0.0 > burn_ratio >= 1.0:
                raise ValueError('burn ratio must be in [0, 1)')
            self._burn = int(self._max_iter * burn_ratio / 100.0)

        if 0.0 >= delta > 1.0:
            raise ValueError('Reward steps (delta) must be in (0, 1)')
        self._delta = delta

    def solve(self, mdp, demos):
        """ Solve the BIRL problem """
        v = ['step', 'r', 'r_mean', 'sample', 'a_ratio']
        trace = Trace(v, save_interval=self._max_iter//2)

        if self._inference == 'PW':
            return self.policy_walk(mdp, demos, trace)
        elif self._inference == 'MAP':
            return self.find_map(mdp, demos, trace)

    def policy_walk(self, mdp, demos, trace):
        """ Find the reward using PolicyWalk """

        self._proposal = PolicyWalkProposal(dim=len(mdp.reward),
                                            delta=self._delta)

        r = self._initialize_reward(mdp.reward.rmax, len(mdp.reward))
        plan_r = self._solve_mdp(mdp, r)

        r_mean = np.array(r)
        for step in tqdm(range(1, self._max_iter+1), desc='PolicyWalk'):
            r_new = self._proposal(r)
            plan_r_new = self._solve_mdp(mdp, r_new, plan_r['V'], plan_r['pi'])
            p_accept = self._acceptance_ratio(mdp, demos, r, r_new,
                                              plan_r['Q'], plan_r_new['Q'])
            if self._rng.uniform() < min([1.0, p_accept])**self._cooling(step):
                r = np.array(r_new)
                plan_r = deepcopy(plan_r_new)

            if step > self._burn:
                r_mean = self._iterative_mean(r_mean, r, step-self._burn)
                trace.record(step=step, r=r, r_mean=r_mean, sample=r_new,
                             a_ratio=p_accept)
        return trace

    def find_map(self, mdp, demos, trace):
        """ Find the reward by direct MAP estimation """

        trace.add_vars(['f', 'r_map'])

        rmax = mdp.reward.rmax
        bounds = tuple((-rmax, rmax) for _ in range(len(mdp.reward)))

        r = self._initialize_reward(mdp.reward.rmax, len(mdp.reward))

        def _callback_optimization(x):
            """ Callback to catch the optimization progress """
            trace.record(r=x)

        self._mdp = mdp  # temporary HACK
        self._demos = demos

        # r is argmax_r p(D|r)p(r)
        res = minimize(fun=self._log_posterior,
                       x0=r,
                       method='L-BFGS-B',
                       jac=False,
                       bounds=bounds,
                       callback=_callback_optimization)

        trace.record(r_map=res.x, f=res.fun)

        return trace

    def _initialize_reward(self, rmax, rdim):
        """ Initialize a reward vector using the prior """
        candidates = np.arange(-rmax, rmax+self._delta, self._delta)
        r = np.zeros(rdim)
        for i in range(rdim):
            r[i] = self._rng.choice(candidates)
        return r

    def _acceptance_ratio(self, mdp, demos, r, r_new, Q_r, Q_r_new):
        ratio = np.prod(np.true_divide(self._prior(r_new), self._prior(r)))
        for traj in demos:
            if traj:
                for (s, a) in traj:
                    rr = np.exp(self._beta * Q_r[a, s]) /\
                            sum(np.exp(self._beta * Q_r[b, s]) for b in mdp.A)
                    rr_new = np.exp(self._beta * Q_r_new[a, s]) /\
                        sum(np.exp(self._beta * Q_r_new[b, s])
                            for b in mdp.A)

                    ratio *= rr_new / rr
        return ratio

    def _log_posterior(self, r):
        """ Reward posterior distribution (unnormalized) """
        plan_r = self._solve_mdp(self._mdp, r)
        llk = self._log_likelihood(plan_r['Q'])
        lp = self._log_prior(r)
        return llk + lp

    def _log_likelihood(self, Q_r):
        """ Evaluate the log likelihood of the demonstrations w.r.t reward """
        llk = 0.0
        M = len(self._demos)
        for traj in self._demos:
            if traj:
                H = len(traj)
                alpha_H = 0.0
                beta_H = 0.0
                for (s, a) in traj:
                    alpha_H += self._beta * Q_r[a, s]
                    beta_Hs = [self._beta * Q_r[b, s] for b in self._mdp.A]
                    beta_H += logsumexp(beta_Hs)

                llk += (alpha_H - beta_H) / float(H)
        llk /= float(M)

        return llk

    def _log_prior(self, r):
        """ Compute log prior probability """
        return self._prior.log_p(r)

    def _iterative_mean(self, r_mean, r_new, step):
        """ Compute the iterative mean of the reward """
        r_mean = [((step - 1) / float(step)) *
                  r_m + 1.0 / step * r for r_m, r in zip(r_mean, r_new)]
        return np.array(r_mean)

    def _cooling(self, step):
        """ Tempering """
        return 5.0 + step / 50.0


########################################################################
# MCMC proposals
########################################################################


class Proposal(object):
    """ Proposal for MCMC sampling """
    def __init__(self, dim):
        self.dim = dim

    @abstractmethod
    def __call__(self, location):
        raise NotImplementedError('abstract')


class PolicyWalkProposal(Proposal):
    """ PolicyWalk MCMC proposal

    Sampling in the hypercube with limits [-rmax, rmax]

    """
    def __init__(self, dim, delta, rmax=1.0, random_state=None):
        super(PolicyWalkProposal, self).__init__(dim)
        self.delta = delta
        self.rmax = rmax
        self.rng = check_random_state(random_state)

    def __call__(self, location):
        sample = np.array(location)
        d = self.rng.choice([-self.delta, self.delta])
        i = self.rng.randint(self.dim)
        if -self.rmax <= sample[i]+d <= self.rmax:
            sample[i] += d
        return sample


########################################################################
# Reward Priors
# ######################################################################


class RewardPrior(six.with_metaclass(ABCMeta, Model)):
    """ Reward prior interface

    The reward prior summarizes information about the reward distribution that
    is available before running the algorithm, i.e. all the relevant domain
    knowledge.

    These distributions are multivariate, i.e. samples are vectors

    """

    def __init__(self, dim):
        if 0 > dim:
            raise ValueError('Reward space dimension must be positive')
        self._dim = dim

    @abstractmethod
    def __call__(self, r):
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def log_p(self, r):
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def sample(self):
        raise NotImplementedError('Abstract method')


class GaussianRewardPrior(RewardPrior):
    """ Gaussian reward prior """
    def __init__(self, dim=1, mean=0.0, sigma=0.5):
        super(GaussianRewardPrior, self).__init__(dim)
        self._cov = np.eye(self._dim) * sigma
        self._mu = np.ones(self._dim) * mean
        # TODO - allow different covariance shapes, and full mean vectors

    def __call__(self, r):
        return multivariate_normal.pdf(r, mean=self._mu, cov=self._cov)

    def log_p(self, r):
        return multivariate_normal.logpdf(r, mean=self._mu, cov=self._cov)

    def sample(self):
        return multivariate_normal.rvs(mean=self._mu, cov=self._cov, size=1)
