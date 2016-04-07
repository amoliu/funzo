"""
BIRL with inference performed using MCMC-type procedures

1. PolicyWalkBIRL, of Ramachandran et. al.
2. ModifiedPolicyWalkBIRL, of Michini and How

"""

from __future__ import division

import numpy as np

from abc import abstractmethod
from tqdm import tqdm
from six.moves import range, zip
from scipy.misc import logsumexp

from .base import BIRL
from ...utils.validation import check_random_state
from ...utils.data_structures import Trace


__all__ = ['PolicyWalkBIRL', 'PolicyWalkProposal']


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
        if -self.rmax < sample[i]+d < self.rmax:
            sample[i] += d
        return sample


#############################################################################


class PolicyWalkBIRL(BIRL):
    """ BIRL with inference done using PolicyWalk MCMC algorithm """
    def __init__(self, mdp, prior, demos, planner, beta, delta=0.2,
                 burn_ratio=0.27, max_iter=100, cooling=False,
                 verbose=4, random_state=None):
        super(PolicyWalkBIRL, self).__init__(mdp, prior, demos, planner, beta)

        if 0 >= max_iter > np.inf:
            raise ValueError('No. of iterations must be in (0, inf)')
        self._max_iter = max_iter

        if 0.0 > burn_ratio >= 1.0:
            raise ValueError('burn ratio must be in [0, 1)')
        self._burn = int(self._max_iter * burn_ratio / 100.0)

        if 0.0 >= delta > 1.0:
            raise ValueError('Reward steps (delta) must be in (0, 1)')
        self._proposal = PolicyWalkProposal(dim=len(self._mdp.reward),
                                            delta=delta)
        self._tempered = cooling
        self._rng = check_random_state(random_state)

    def run(self, **kwargs):
        """ Run the BIRL solver to find the reward function """
        v = ['step', 'r', 'r_mean', 'sample', 'a_ratio', 'Q_r', 'log_p']
        trace = Trace(v, save_interval=self._max_iter//2)

        r = self._initialize_reward()
        r_mean = np.array(r)

        Q_r, llk_r = self._log_likelihood(r)
        log_p_r = llk_r + self._log_prior(r)

        for step in tqdm(range(1, self._max_iter+1), desc='PolicyWalk'):
            r_new = self._proposal(r)
            Q_r_new, llk_r_new = self._log_likelihood(r_new)
            log_p_r_new = llk_r_new + self._log_prior(r_new)

            p_accept = log_p_r_new / log_p_r
            if self._rng.uniform() < min([1.0, p_accept]):
                r = np.array(r_new)
                log_p_r = log_p_r_new

            if step > self._burn:
                r_mean = self._iterative_mean(r_mean, r, step-self._burn)
                trace.record(step=step, r=r, r_mean=r_mean, sample=r_new,
                             a_ratio=p_accept, Q_r=Q_r_new, log_p=log_p_r_new)

        return trace

    def _initialize_reward(self):
        """ Initialize a reward vector using the prior """
        return self._prior.sample()

    def _log_likelihood(self, r):
        """ Evaluate the log likelihood of the demonstrations w.r.t reward """
        self._mdp.reward.update_parameters(reward=r)
        Q_r = self._planner(self._mdp)['Q']

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

        return Q_r, llk

    def _log_prior(self, r):
        """ Compute log prior probability """
        return self._prior.log_p(r)

    def _iterative_mean(self, r_mean, r_new, iteration):
        """ Compute the iterative mean of the reward """
        r_mean = [((iteration - 1) / float(iteration)) *
                  r_m + 1.0 / iteration * r for r_m, r in zip(r_mean, r_new)]
        return np.array(r_mean)

    def cooling(self, step):
        """ Cooling schedule """
        return 5.0 + step / 50.0
