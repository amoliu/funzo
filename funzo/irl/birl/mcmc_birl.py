"""
BIRL with inference performed using MCMC-type procedures

1. PolicyWalkBIRL, of Ramachandran et. al.
2. ModifiedPolicyWalkBIRL, of Michini and How

"""

from __future__ import division

import numpy as np

from copy import deepcopy
from scipy.misc import logsumexp

from .base import BIRL
from ...utils.validation import check_random_state
from ...utils.data_structures import Trace


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
        changed = False
        while not changed:
            d = self.rng.choice([-self.delta, 0, self.delta])
            i = self.rng.randint(self.dim)
            if -self.rmax <= sample[i]+d <= self.rmax:
                sample[i] += d
                changed = True

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
    def __init__(self, mdp, prior, demos, planner, beta,
                 burn_ratio=0.27, max_iter=100, verbose=4):
        super(PolicyWalkBIRL, self).__init__(mdp, prior, demos, planner, beta)

        assert 0 < max_iter < np.inf, 'iterations must be in (0, inf)'
        self._max_iter = max_iter

        assert 0.0 <= burn_ratio < 1.0, 'burn ratio must be in [0, 1)'
        self._burn_point = int(self._max_iter * burn_ratio / 100.0)

    def run(self, **kwargs):
        r = self._initialize_reward(random_state=None)
        r_mean = deepcopy(r)

        # get policy and posterior for current reward
        Q_r, log_p_r = self._compute_log_posterior(r)

        proposal = PolicyWalkProposal(dim=3, delta=0.2)

        trace = Trace(save_interval=self._max_iter//10)
        step = 1
        while step < self._max_iter:
            r_new = proposal(r)
            Q_r_new, log_p_r_new = self._compute_log_posterior(r_new)

            # Add line 3 (c) from Ramachandran (only for AL), not needed??

            next_r, pr = pw_metrop_select(r, r_new, log_p_r, log_p_r_new)

            r = deepcopy(next_r)
            r_mean = self._iterative_reward_mean(r_mean, r, step)

            trace.record(r, r_new, step, pr, Q_r_new, log_p_r_new)

            step += 1

        return trace, r_mean

    def _initialize_reward(self, random_state=None):
        """ Initialize a reward vector using the prior """
        rng = check_random_state(random_state)
        r = rng.rand(len(self._mdp.reward))
        return self._prior(r)

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

    def _iterative_reward_mean(self, r_mean, r_new, iteration):
        """ Compute the iterative mean of the reward """
        r_mean = [((iteration - 1) / float(iteration)) *
                  m_r + 1.0 / iteration * r for m_r, r in zip(r_mean, r_new)]

        return np.array(r_mean)
