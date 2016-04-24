"""
The :mod:`funzo.domains` module contains reinforcement learning environments
"""

from .base import Domain, model_domain

from .geometry import discretize_space, distance_to_segment, edist

from .gridworld import GAction, GState, GTransition, GReward, GRewardLFA
from .gridworld import GridWorldMDP, GridWorld

from .puddleworld import PWState, PWAction, PWTransition, PuddleReward
from .puddleworld import PuddleWorldMDP, PuddleWorld


__all__ = [
    'Domain', 'model_domain',
    #
    'discretize_space', 'distance_to_segment', 'edist',
    #
    'GAction', 'GState', 'GTransition', 'GReward', 'GRewardLFA',
    'GridWorldMDP', 'GridWorld',
    #
    'PWState', 'PWAction', 'PWTransition', 'PuddleReward',
    'PuddleWorldMDP', 'PuddleWorld',
]
