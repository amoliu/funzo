"""
BIRL with inference performed using MCMC-type procedures

1. PolicyWalkBIRL, of Ramachandran et. al.
2. ModifiedPolicyWalkBIRL, of Michini and How

"""

from __future__ import division

import numpy as np

from tqdm import tqdm
from copy import deepcopy
from six.moves import range, zip
from scipy.misc import logsumexp

from .base import BIRL
from ...utils.validation import check_random_state
from ...utils.data_structures import Trace


__all__ = ['PolicyWalkBIRL']


class Proposal(object):
    """ Proposal for MCMC sampling """
    def __init__(self, dim):
        self.dim = dim


class PolicyWalkProposal(Proposal):
    """ PolicyWalk MCMC proposal

    Sampling in the hypercube with limits [-rmax, rmax]

    """
    def __init__(self, dim, delta, rmax=1.0, random_state=None):
        super(PolicyWalkProposal, self).__init__(dim)
        self.delta = delta
        self.rmax = rmax
        self.rng = check_random_state(random_state)

    def __call__(self, loc):
        sample = np.asarray(loc)

        d = self.rng.choice([-self.delta, self.delta])
        i = self.rng.randint(self.dim)
        if -self.rmax < sample[i]+d < self.rmax:
            sample[i] += d

        return sample


#############################################################################


def pw_metrop_select(r, r_new, log_p_r, log_p_r_new):
    """ Metropolis-Hastings type select function for accepting samples

    Parameters
    -----------
    r, r_new : array-like
        Reward parameter vectors
    log_p_r, log_p_r_new : float
        Unnormalized Log of posterior probability p(r | demonstration)

    Returns
    --------
    x : array-like
        Reward parameter vector
    p_ratio : float
        Acceptance ratio

    """
    p_ratio = log_p_r_new / log_p_r
    if np.isfinite(p_ratio) and np.log(np.random.uniform()) < p_ratio:
        return r_new, p_ratio
    else:
        return r, p_ratio


#############################################################################


class PolicyWalkBIRL(BIRL):
    """ BIRL with inference done using PolicyWalk MCMC algorithm """
    def __init__(self, mdp, prior, demos, planner, beta, delta=0.2,
                 burn_ratio=0.27, max_iter=100, verbose=4):
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
                                            delta=delta,
                                            rmax=1.0)

    def run(self, **kwargs):
        r = self._initialize_reward(random_state=None)
        r_mean = deepcopy(r)

        # get policy and posterior for current reward
        Q_r, log_p_r = self._compute_log_posterior(r)
        trace = Trace(save_interval=self._max_iter//5)

        for step in tqdm(range(1, self._max_iter+1)):
            r_new = self._proposal(r)
            Q_r_new, log_p_r_new = self._compute_log_posterior(r_new)

            # Add line 3 (c) from Ramachandran (only for AL), not needed??
            # Cooling?

            next_r, pr = pw_metrop_select(r, r_new, log_p_r, log_p_r_new)
            r = deepcopy(next_r)

            if step > self._burn:
                r_mean = self._iterative_mean(r_mean, r, step-self._burn)

            trace.record(step, r, r_new, pr, Q_r_new, log_p_r_new)
            step += 1

        return trace, r_mean

    def _initialize_reward(self, random_state=None):
        """ Initialize a reward vector using the prior """
        # rng = check_random_state(random_state)
        # # todo - ensure we sample uniformly from hypercube of -rmax, rmax
        # r = rng.rand(len(self._mdp.reward))
        # return self._prior(r)

        rmax = self._mdp.reward.rmax
        tp = [-rmax + i * 0.2 for i in xrange(int(rmax * 2 / 0.2 + 1))]
        Theta = []
        for i in range(len(self._mdp.reward)):
            Theta.append(tp[np.random.randint(int(rmax * 2 / 0.2 + 1))])
        return Theta

    def _compute_log_posterior(self, r):
        """ Evaluate the log posterior probability w.r.t reward """
        # solve MDP to get Q_r, Q_r_new
        self._mdp.reward.update_parameters(reward=r)
        Q_r = self._planner(self._mdp)['Q']

        M = len(self._demos)
        llk = 0.0
        for traj in self._demos:
            if traj:
                H = len(traj)
                alpha_H = 0.0
                for (s, a) in traj:
                    alpha_H += self._beta * Q_r[a, s]
                    beta_Hs = list()
                    for b in self._mdp.A:
                        beta_Hs.append(self._beta * Q_r[b, s])
                    beta_H = logsumexp(beta_Hs)

                llk += (alpha_H - beta_H) / float(H+1)
        llk /= float(M)

        # compute log priors
        log_prior = np.sum(self._prior.log_p(r))

        # compute full posterior
        log_p = llk + log_prior

        return Q_r, log_p

    def _iterative_mean(self, r_mean, r_new, iteration):
        """ Compute the iterative mean of the reward """
        r_mean = [((iteration - 1) / float(iteration)) *
                  r_m + 1.0 / iteration * r for r_m, r in zip(r_mean, r_new)]

        return np.array(r_mean)
