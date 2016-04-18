"""
The :mod:`funzo.planners` module contains reinforcement learning *planning*
algorithms
"""

from .base import Planner

from .dp import PolicyIteration, ValueIteration


__all__ = [
    'Planner',
    #
    'PolicyIteration', 'ValueIteration',
    #
]
