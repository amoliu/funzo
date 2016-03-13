"""
BIRL with inference performed using MCMC-type procedures

1. PolicyWalkBIRL, of Ramachandran et. al.
2. ModifiedPolicyWalkBIRL, of Michini and How

"""

from __future__ import division

import numpy as np

from .base import BIRL
from ...utils.validation import check_random_state


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


class PolicyWalkBIRL(BIRL):
    """ BIRL with inference done using PolicyWalk MCMC algorithm """
    def __init__(self, mdp, prior, demos, planner, beta,
                 burn_ratio=0.27, max_iter=100, verbose=4):
        super(PolicyWalkBIRL, self).__init__(mdp, prior, demos, planner, beta)

        assert 0 < max_iter < np.inf, 'iterations must be in (0, inf)'
        self._max_iter = max_iter

        assert 0.0 <= burn_ratio < 1.0, 'burn ratio must be in [0, 1)'
        self._burn_point = int(self._max_iter * burn_ratio / 100.0)

        self._trace = dict()

    def run(self, **kwargs):
        step = 1
        while step < self._max_iter:
            pass

    def _initialize_reward(self, random_state=None):
        """ Initialize a reward vector using the prior """
        rng = check_random_state(random_state)
        r = rng.rand(len(self._mdp.reward))
        return self._prior(r)
