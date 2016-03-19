"""
PuddleWorld Domain
===================

A continuous environment in which the agent has to avoid getting into various puddles on the way to the goal.

"""

from __future__ import division

import numpy as np

from six.moves import range
from collections import Iterable
from matplotlib.patches import Rectangle

from .base import Domain

from ..models.mdp import MDP
from ..models.mdp import TabularRewardFunction, LinearRewardFunction
from ..models.mdp import MDPTransition, MDPState, MDPAction

from ..utils.validation import check_random_state
from ..utils.geometry import distance_to_segment, edist


class Agent(object):
    """ A agent object """
    def __init__(self, position, orientation, visual, **kwargs):
        self.position = position
        self.orientation = orientation
        self.visual = visual


class Puddle(object):
    """ A puddle in a continous puddle world

    Represented by combinations of a line and semi-circles at each end,

    Parameters
    -----------
    x1, x2, y1, y2 : float
        Coordinates of the puddle midline
    radius : float
        Thickness/breadth of the puddle in all directions

    Attributes
    -----------
    start : array-like
        1D numpy array with the start of the line at the puddle center line
    end: array-like
        1D numpy array with the end of the line at the puddle center line
    radius: float
        Thickness/breadth of the puddle in all directions

    """
    PUDDLE_COST = 100

    def __init__(self, x1, y1, x2, y2, radius, **kwargs):
        assert x1 >= 0 and x1 <= 1, 'Puddle coordinates must be in [0, 1]'
        assert x2 >= 0 and x2 <= 1, 'Puddle coordinates must be in [0, 1]'
        assert y1 >= 0 and y1 <= 1, 'Puddle coordinates must be in [0, 1]'
        assert y2 >= 0 and y2 <= 1, 'Puddle coordinates must be in [0, 1]'
        assert radius > 0, 'Puddle radius must be > 0'
        self.start = np.array([x1, y1])
        self.end = np.array([x2, y2])
        self.radius = radius

    def cost(self, x, y):
        dist_puddle, inside = distance_to_segment((x, y), self.start,
                                                  self.end)
        if inside:
            if dist_puddle < self.radius:
                return -self.PUDDLE_COST * (self.radius - dist_puddle)
        else:
            d = min(edist((x, y), self.start), edist((x, y), self.end))
            if d < self.radius:
                return -self.PUDDLE_COST * (self.radius - d)
        return 0.0

    @property
    def location(self):
        return self.start[0], self.start[1], self.end[0], self.end[1]

    @property
    def length(self):
        return edist(self.start, self.end)
