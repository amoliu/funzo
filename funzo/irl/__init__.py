"""
The :mod:`funzo.irl` module contains inverse reinforcement learning algorithms
"""

from .irl_base import IRLSolver
from .irl_base import Loss, PolicyLoss, RewardLoss

from . import birl

__all__ = [
    'IRLSolver',
    #
    'Loss', 'PolicyLoss', 'RewardLoss',
    #
    'birl',
]
