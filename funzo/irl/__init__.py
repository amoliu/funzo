"""
The :mod:`funzo.irl` module contains inverse reinforcement learning algorithms
"""

from .base import IRLSolver
from .base import Loss, PolicyLoss, RewardLoss

from .birl import *

__all__ = [
    'IRLSolver',
    #
    'Loss', 'PolicyLoss', 'RewardLoss',
    #
]
