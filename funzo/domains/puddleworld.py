"""
PuddleWorld Domain
===================

A continuous environment in which the agent has to avoid getting into various
puddles on the way to the goal.

"""

from __future__ import division

import numpy as np

from six.moves import range
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Wedge, Circle

from .base import Domain

from ..models.mdp import MDP
from ..models.mdp import TabularRewardFunction
from ..models.mdp import MDPTransition, MDPState, MDPAction

from ..utils.geometry import distance_to_segment, edist

from .geometry import discretize_space


__all__ = [
    'PuddleWorld',
    'PuddleWorldMDP',
    'PuddleReward',
    'PWTransition',
    'PWAction',
    'PWState',
]


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
        p_cost = np.sum(p.cost(state_.location[0], state_.location[1])
                        for p in self._domain.puddles)
        return -self._sr + p_cost

    def __len__(self):
        return 1

#############################################################################


class PWState(MDPState):
    """ PuddleWorld state """
    def __init__(self, state_id, location):
        super(PWState, self).__init__(state_id)
        x_check = 0.0 <= location[0] <= 1.0
        y_check = 0.0 <= location[1] <= 1.0
        if not x_check or not y_check:
            raise ValueError('Puddle state locations must be in [0, 1]')
        self._s = location

    @property
    def location(self):
        return np.array(self._s)

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return (self.location[0], self.location[1]).__hash__()

    def __str__(self):
        return 'State {}:({}, {})'.format(self.id, self._s[0], self._s[1])

    def __repr__(self):
        return self.__str__()


class PWAction(MDPAction):
    """ PuddleWorld action """
    def __init__(self, action_id, direction, step=0.05):
        super(PWAction, self).__init__(action_id)
        if not 0.0 < step < 1.0:
            raise ValueError('PW Action step must be in (0, 1)')
        self._a = (0.0, step)
        if direction == 'LEFT':
            self._a = (-step, 0.0)
        elif direction == 'RIGHT':
            self._a = (step, 0.0)
        elif direction == 'DOWN':
            self._a = (0.0, -step)
        self.name = direction

    @property
    def direction(self):
        return np.array(self._a)

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return (self.direction[0], self.direction[1]).__hash__()

    def __str__(self):
        return 'Action {}:({}, {})'.format(self.id, self._a[0], self._a[1])

    def __repr__(self):
        return self.__str__()


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

        # noise = np.random.normal(0.0, scale=0.01)
        # TODO - fixme (the noise should only be in the chosen direction)
        action_vector = action_.direction
        next_state = state_.location + action_vector

        # check if in world,, find its id
        if self._domain.in_domain(next_state):
            ns_id = self._domain.find_state(next_state[0], next_state[1])
            if ns_id is not None:
                return [(1.0, ns_id)]
        return [(1.0, state)]


#############################################################################


class PuddleWorld(Domain):
    """ PuddleWorld domain """

    def __init__(self, start, resolution=0.1):
        self._start = start

        self.states = dict()
        state_id = 0

        a, b = discretize_space((0, 1, resolution), (0, 1, resolution))
        self.w, self.h = a.shape
        for i in range(self.w):
            for j in range(self.h):
                x, y = a[i, j] + resolution/2., b[i, j] + resolution/2.
                self.states[state_id] = PWState(state_id, (x, y))
                state_id += 1

        self.actions = {
            0: PWAction(0, 'UP', step=resolution),
            1: PWAction(1, 'DOWN', step=resolution),
            2: PWAction(2, 'LEFT', step=resolution),
            3: PWAction(3, 'RIGHT', step=resolution)
        }

        self.puddles = list()
        self.puddles.append(Puddle(0.1, 0.75, 0.45, 0.75, 0.1))
        self.puddles.append(Puddle(0.45, 0.4, 0.45, 0.8, 0.1))

    def terminal(self, state):
        """ Check if a state is terminal"""
        state_ = self.states[state]
        return state_[0] > 0.95 and state_[1] > 0.95

    def in_domain(self, location):
        return 0.0 < location[0] < 1.0 and 0.0 < location[1] < 1.0

    def visualize(self, ax, **kwargs):
        ax = self._setup_visuals(ax)

        if 'policy' in kwargs:
            self.show_policy(ax, kwargs['policy'])

        return ax

    def find_state(self, x, y):
        for s in self.states:
            if edist(self.states[s].location, (x, y)) < 1e-07:
                return s
        return None

    @property
    def shape(self):
        return self.w, self.h

    def _setup_visuals(self, ax):
        """Setup visual elements
        """
        # Main rectangle showing the environment
        ax.add_artist(Rectangle((0, 0), width=1, height=1, color='c',
                                zorder=0, ec='k', lw=8, fill=False))

        # draw goal region
        points = [[1, 1], [1, 0.95], [0.95, 1]]
        goal_polygon = plt.Polygon(points, color='green')
        ax.add_patch(goal_polygon)

        # draw puddles
        x1 = self.puddles[0].start[0]
        y1 = self.puddles[0].start[1]-0.05
        width = self.puddles[0].length
        height = self.puddles[1].length
        pd1 = Rectangle((x1, y1), height=0.1, width=width,
                        color='brown', alpha=0.7, aa=True, lw=0)
        ax.add_artist(pd1)
        ax.add_artist(Wedge(self.puddles[0].start, 0.05, 90, 270,
                            fc='brown', alpha=0.7, aa=True, lw=0))
        ax.add_artist(Wedge(self.puddles[0].end, 0.05, 270, 90,
                            fc='brown', alpha=0.7, aa=True, lw=0))

        x2 = self.puddles[1].start[0]-0.05
        y2 = self.puddles[1].start[1]
        pd2 = Rectangle((x2, y2), width=0.1, height=height,
                        color='brown', alpha=0.7)
        ax.add_artist(pd2)
        ax.add_artist(Wedge(self.puddles[1].start, 0.05, 180, 360,
                            fc='brown', alpha=0.7, aa=True, lw=0))
        ax.add_artist(Wedge(self.puddles[1].end, 0.05, 0, 180,
                            fc='brown', alpha=0.7, aa=True, lw=0))

        # draw the agent at initial pose
        robot_start = (0.3, 0.65)
        robot_visual = Circle(robot_start, 0.01, fc='b', ec='k', zorder=3)
        ax.add_artist(robot_visual)
        self.robot = Agent(position=robot_start, orientation=(1, 1),
                           visual=robot_visual)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xticks([])
        ax.set_yticks([])

        return ax

    def show_policy(self, ax, policy):
        """ Show a policy on the interface """
        if len(policy) != len(self.states):
            raise ValueError('Policy not compatible with state dimensions')
        for s in range(policy.shape[0]):
            a = policy[s]
            if self.actions[a].name == 'RIGHT':
                text = '$\\rightarrow$'
            elif self.actions[a].name == 'UP':
                text = '$\\uparrow$'
            elif self.actions[a].name == 'LEFT':
                text = '$\\leftarrow$'
            elif self.actions[a].name == 'DOWN':
                text = '$\\downarrow$'
            else:
                text = 'G'
            ss = self.states[s]
            ax.text((ss.location[0] * 1), (ss.location[1] * 1),
                    text, ha="center", size=10)
        return ax


class PuddleWorldMDP(MDP):
    """ PuddleWorld MDP representing the decision making process """
    def __init__(self, domain, reward, transition, discount=0.9):
        super(PuddleWorldMDP, self).__init__(domain,
                                             reward,
                                             transition,
                                             discount)

    @property
    def S(self):
        """ States of the MDP in an indexable container """
        return self._domain.states.keys()

    @property
    def A(self):
        """ Actions of the MDP in an indexable container """
        return self._domain.actions.keys()

    def actions(self, state):
        return self._domain.actions.keys()
