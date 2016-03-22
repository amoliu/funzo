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


__all__ = ['PolicyWalkBIRL', 'PolicyWalkProposal']


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


def pw_metrop_select(r, r_new, log_p_r, log_p_r_new, tempering=1.0):
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
    p_ratio = (log_p_r_new / log_p_r)**tempering
    if np.isfinite(p_ratio) and np.log(np.random.uniform()) < p_ratio:
        return r_new, p_ratio
    else:
        return r, p_ratio


#############################################################################


class PolicyWalkBIRL(BIRL):
    """ BIRL with inference done using PolicyWalk MCMC algorithm """
    def __init__(self, mdp, prior, demos, planner, beta, delta=0.2,
                 burn_ratio=0.27, max_iter=100, cooling=False, verbose=4):
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

    def run(self, **kwargs):
        r = self._initialize_reward(random_state=None)
        r_mean = deepcopy(r)

        mr = list()
        mr.append(r_mean)

        trace = Trace(save_interval=self._max_iter//2)

        Q_r_old, llk_old = self._compute_llk(r)
        log_prior_old = self._log_prior(r)
        log_p_r_old = llk_old + log_prior_old

        for step in tqdm(range(1, self._max_iter+1), desc='PW'):
            r_new = self._proposal(r)
            Q_r_new, llk_new = self._compute_llk(r_new)
            log_prior_new = self._log_prior(r_new)

            # compute full posterior distribution (unnormalized)
            log_p_r_new = llk_new + log_prior_new

            # ratio of joint vs conditionals for correctness check
            # print(log_p_r_new/log_p_r_old, llk_new/llk_old)

            next_r, pr = pw_metrop_select(r, r_new,
                                          log_p_r_old, log_p_r_new,
                                          self.tempering(step))
            r = deepcopy(next_r)
            print(r)

            if step > self._burn:
                r_mean = self._iterative_mean(r_mean, r, step-self._burn)

            trace.record(step, r, r_new, pr, Q_r_new, log_p_r_new)
            mr.append(r_mean)
            step += 1

        return trace, mr

    def _initialize_reward(self, random_state=None):
        """ Initialize a reward vector using the prior """
        return self._prior.sample(dim=len(self._mdp.reward))
        # rng = check_random_state(random_state)
        # rmax = self._mdp.reward.rmax
        # r = rng.uniform(low=-rmax, high=rmax, size=len(self._mdp.reward))
        # return self._prior(r)

    def _compute_llk(self, r):
        """ Evaluate the log likelihood of the demonstrations w.r.t reward """
        # solve MDP to get Q_r, Q_r_new
        self._mdp.reward.update_parameters(reward=r)
        Q_r = self._planner(self._mdp)['Q']

        # log likelihood of the data
        llk = 0.0
        M = len(self._demos)
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

        return Q_r, llk

    def _log_prior(self, r):
        """ Compute log prior probability """
        return np.sum(self._prior.log_p(r))

    def _iterative_mean(self, r_mean, r_new, iteration):
        """ Compute the iterative mean of the reward """
        r_mean = [((iteration - 1) / float(iteration)) *
                  r_m + 1.0 / iteration * r for r_m, r in zip(r_mean, r_new)]
        return np.array(r_mean)

    def tempering(self, step):
        """ Cooling schedule """
        return 5.0 + step / 50.0
