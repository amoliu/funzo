"""
PuddleWorld Domain
===================

A continuous environment in which the agent has to avoid getting into various
puddles on the way to the goal.

"""

from __future__ import division

import numpy as np

from six.moves import range
from collections import Iterable
from matplotlib.patches import Rectangle

from .base import Domain

from ..models.mdp import MDP
from ..models.mdp import TabularRewardFunction
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
    """ A puddle in a continuous puddle world

    Represented by combinations of a line and semi-circles at each end,

    Parameters
    -----------
    x1, x2, y1, y2 : float
        Coordinates of the puddle mid-line
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
    PUDDLE_COST = 400

    def __init__(self, x1, y1, x2, y2, radius, **kwargs):
        if radius <= 0:
            raise ValueError('Puddle radius must be > 0')
        self.radius = radius
        self.start = np.array([self._validate(x1), self._validate(y1)])
        self.end = np.array([self._validate(x2), self._validate(y2)])

    def cost(self, x, y):
        d, inside = distance_to_segment((x, y), self.start, self.end)
        if inside:
            if d < self.radius:
                return -self.PUDDLE_COST * (self.radius - d)
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

    def _validate(self, x):
        if not 0.0 <= x <= 1.0:
            raise ValueError('Puddle coordinates must be in [0, 1]')
        return x


########################################################################


class PuddleReward(TabularRewardFunction):
    """ Reward function for the puddle """
    def __init__(self, domain, rmax=1.0, step_reward=0.1):
        super(PuddleReward, self).__init__(domain, rmax)
        self._sr = step_reward

    def __call__(self, state, action):
        state_ = self._domain.states[state]
        p_cost = np.sum(p.cost(state_[0], state_[1])
                        for p in self._domain.puddles)
        return -self._sr + p_cost

    def __len__(self):
        return 1

#############################################################################


class PWState(MDPState):
    """ PuddleWorld state """
    def __init__(self, state_id, location):
        super(PWState, self).__init__(state_id)
        self.location = location

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return (self.location[0], self.location[1]).__hash__()


class PWAction(MDPAction):
    """ PuddleWorld action """
    def __init__(self, action_id, direction):
        super(PWAction, self).__init__(action_id)
        self.direction = direction

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return (self.direction[0], self.direction[1]).__hash__()


#############################################################################


class PWTransition(MDPTransition):
    """ PuddleWorld transition function """
    def __init__(self, domain):
        super(PWTransition, self).__init__(domain)

    def __call__(self, state, action, **kwargs):
        """ Evaluate transition function

        Returns
        --------
        n_s : array-like
            A list of all possible next states [(prob, state)]

        """
        state_ = self._domain.states[state]
        action_ = self._domain.actions[action]
        action_vector = np.array(action_.direction) *\
            np.random.normal(0.0, scale=0.01)

        next_state = np.array(state_.location) + action_vector

        # check if in world,, find its id
        ns_id = self._domain.state_map[next_state]

        return (1.0, ns_id)
