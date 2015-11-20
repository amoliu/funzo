"""
GridWorld Domain
"""

from __future__ import division

from collections import Iterable
from matplotlib.patches import Rectangle

import six
import numpy as np

from ..base import Model
from ..models.domain import Domain
from ..models.mdp import MDP, MDPReward, MDPTransition, MDPState, MDPAction


class GridReward(MDPReward):
    """ Grid world MDP reward function """
    def __init__(self, domain):
        super(GridReward, self).__init__(domain)

    def __call__(self, state, action):
        return 1.0

    def __len__():
        return len(self._domain.S)


class GTransition(MDPTransition):
    """ Grirdworld MDP controller """
    def __init__(self, domain, noise_level=0):
        super(GTransition, self).__init__(domain)
        self._noise_level = noise_level

    def __call__(self, state, action, **kwargs):
        """ Transition

        Returns
        --------
        A list of all possible next states [(prob, state)]

        """
        state_ = self._domain.S[state]
        action_ = self._domain.A[action]
        return [(0.8, self._move(state_, action_)),
                (0.1, self._move(state_, self._right(action_,
                                                     self.A.values()))),
                (0.1, self._move(state_, self._left(action_,
                                                    self.A.values())))]

    def _move(self, state, direction):
        """ Return the state that results from going in this direction. Stays
        in the same state if action os leading to go outside the world or to
        obstacles
        """
        ns = (state[0]+direction[0], state[1]+direction[1])
        if not self._world.grid.valid_cell(ns):
            return self._world.state_map[state]
        if self._world.grid.blocked(ns):
            return self.state_map[state]
        return self._world.state_map[ns]

    def _heading(self, heading, inc, directions):
        return directions[(directions.index(heading) + inc) % len(directions)]

    def _right(self, heading, directions):
        return _heading(heading, -1, directions)

    def _left(self, heading, directions):
        return _heading(heading, +1, directions)


class GState(MDPState):
    """ Gridworld state """
    def __init__(self, cell, status='free'):
        self.cell = cell
        self.status = status

    def __hash__(self):
        return (self.cell[0], self.cell[1]).__hash__()

    def __eq__(self, other):
        return np.hypot(self.cell[0] - other.cell[0],
                        self.cell[1] - other.cell[1]) < 1e-05

    def __str__(self):
        return '({}, {})'.format(self.cell[0], self.cell[1])

    def __repr__(self):
        return self.__str__()


class GAction(MDPAction):
    """ Grirdworld action """
    def __init__(self, direction):
        self.direction = direction

    def __hash__(self):
        return (self.direction[0], self.direction[1]).__hash__()

    def __eq__(self, other):
        try:
            return all(self.direction == other.direction)
        except Exception:
            return False

    def __str__(self):
        return '[{}, {}]'.format(self.direction[0], self.direction[1])

    def __repr__(self):
        return self.__str__()


class GridWorld(Domain, MDP):
    """ GridWorld domain

    A discrete world with cells (free, obstacles, and goal). The main task
    is to find a path from any start cell to a goal cell.

    """

    def __init__(self, gmap):
        gr = GridReward(domain=self)
        gt = GTransition(domain=self)

        MDP.__init__(self, discount=0.9, reward=gr, transition=gt)

        self._gmap = np.asarray(gmap)
        assert self._gmap.ndim == 2, '`gmap` must be a two array'
        self._initialize(self._gmap)

    def _initialize(self, gmap):
        self._height, self._width = gmap.shape
        self._states = set()

        for i in range(self._height):
            for j in range(self._width):
                print(i, j, gmap[i, j])
                if gmap[i, j] == 1:
                    self._states.add(GState((i, j), BLOCKED))
                if gmap[i, j] == 2:
                    self._states.add(GState((i, j), TERMINAL))
                else:
                    self._states.add(GState((i, j), FREE))

        self._actions = set((GAction((1, 0)),
                             GAction((0, 1)),
                             GAction((-1, 0)),
                             GAction((0, -1))))

    @property
    def S(self):
        """ States of the MDP in an indexable container """
        return self._states

    @property
    def A(self):
        """ Actions of the MDP in an indexable container """
        return self._actions

    def terminal(self, state):
        """ Check if a state is terminal"""
        return state.status == 'terminal'

    def visualize(self, ax, **kwargs):
        if 'show_policy' in kwargs and 'policy' in kwargs:
            print('showing policy')

        ax.imshow(self._gmap, interpolation='nearest',
                  cmap='Paired', vmin=0, vmax=2)
        ax.set_xticks([])
        ax.set_yticks([])

        return ax

    def _show_policy(self, ax, **kwargs):
        pass


# Cell status
FREE = 'free'
BLOCKED = 'blocked'
TERMINAL = 'terminal'
