"""
BIRL with inference performed using MCMC-type procedures

1. PolicyWalkBIRL, of Ramachandran et. al.
2. ModifiedPolicyWalkBIRL, of Michini and How

"""

from __future__ import division

import numpy as np
import pymc3 as pm

from .base import BIRL
from ...utils.validation import check_random_state


class GridWalkProposal(pm.Proposal):
    """ Sampling proposal corresponding to GridWalk


    """
    def __init__(self, s, delta=0.1):
        super(GridWalkProposal, self).__init__(s)
        self.delta = delta

    def __call__(self):
        index = np.random.randint(len(self.s))
        change = 2
        return


class PolicyWalk(pm.Metropolis):
    """ PolicyWalk Sampler """
    def __init__(self, r_init, model=None, **kwargs):
        proposal_dist = GridWalkProposal(r_init)
        super(PolicyWalk, self).__init__(vars=None,
                                         S=None,
                                         proposal_dist=proposal_dist,
                                         scaling=1.0,
                                         tune=True,
                                         tune_interval=100,
                                         model=model,
                                         kwargs)

    def astep(self, r0):
        pass
        # define the delta step
        # define the components of the metropolis select

    @staticmethod
    def competence(var):
        if var.dtype in continuous_types:
            return Competence.compatible
        return Competence.incompatible


#############################################################################


class PolicyWalkBIRL(BIRL):
    """ BIRL with inference done using PolicyWalk MCMC algorithm """
    def __init__(self, mdp, prior, demos, planner, beta,
                 max_iter=50, verbose=4):
        super(PolicyWalkBIRL, self).__init__(mdp, prior, demos, planner, beta)
        self._max_iter = max_iter

        self._trace = dict()

    def run(self, **kwargs):
        pass

    def _initialize_reward(self, random_state=None):
        """ Initialize a reward vector using the prior """
        rng = check_random_state(random_state)
        r = rng.rand(len(self._mdp.reward))
        return self._prior(r)
