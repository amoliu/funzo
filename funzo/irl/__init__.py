"""
The :mod:`funzo.irl` module contains inverse reinforcement learning algorithms
"""

from .base import IRLSolver
from .base import Loss, PolicyLoss, RewardLoss

from .birl import BIRL
from .birl import Proposal, PolicyWalkProposal
from .birl import RewardPrior, GaussianRewardPrior


__all__ = [
    'IRLSolver',
    #
    'Loss', 'PolicyLoss', 'RewardLoss',
    #
    'BIRL',
    'Proposal', 'PolicyWalkProposal',
    'RewardPrior', 'GaussianRewardPrior',
    #
]
