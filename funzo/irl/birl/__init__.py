"""
The :mod:`funzo.irl.birl` module contains BIRL algorithms
"""

from .base import BIRLBase

from .priors import RewardPriorBase, GaussianRewardPrior, UniformRewardPrior

from .mcmc_birl import PolicaWalkBIRL, PolicyWalkProposal

from .opt_birl import MAPBIRL


__all__ = [
    'BIRLBase',
    #
    'RewardPriorBase', 'GaussianRewardPrior', 'UniformRewardPrior',
    #
    'PolicaWalkBIRL', 'PolicyWalkProposal',
    #
    'MAPBIRL',
]
