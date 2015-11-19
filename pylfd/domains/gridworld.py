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
from ..models.mdp import MDP, MDPReward, MDPTransition


class Grid2D(Model):
    """ A 2D grid representation

    Represents a rectangular grid map. The map consists of `nrows` x `ncols`
    cells (squares). The cell value denotes different properties such as:

        1: Blocked (by obstacles).
        0: Empty (free)
        2: Goal (absorbing state in SSP MDP sense)

    """

    def __init__(self, nrows, ncols):
        self._cols = None
        self._rows = None
        self.rows = nrows
        self.cols = ncols
        self._map = np.zeros(shape=(nrows, ncols), dtype=np.int)

    def block_cell(self, c):
        self._map[c[0], c[1]] = 1

    def unblock_cell(self, c):
        self._map[c[0], c[1]] = 0

    def blocked(self, c):
        """ Check is a cell is blocked """
        return self._map[c[0], c[1]] == 1

    def valid_cell(self, cell):
        """
        Check all vallidity in the general sense, i.e. not outside the grid
        """
        # and self._map[r][c] == 0):
        r, c = cell[0],  cell[1]
        if 0 <= r <= self._rows - 1 and 0 <= c <= self._cols - 1:
            return True
        return False

    @property
    def rows(self):
        return self._rows

    @rows.setter
    def rows(self, value):
        self._rows = self._check_dim(value, 'rows')

    @property
    def cols(self):
        return self._cols

    @cols.setter
    def cols(self, value):
        self._cols = self._check_dim(value, 'cols')

    @property
    def cells(self):
        return list((i, j) for i in range(self._rows)
                    for j in range(self._cols))

    def __repr__(self):
        """ Print the map to stdout in ASCII """
        for row in range(self._rows):
            for col in range(self._cols):
                print("%s" % ('[X]' if self._map[row, col] == 1 else '[ ]')),
            print('')

    def __str__(self):
        return self.__repr__

    def _check_dim(self, value, dim):
        value = int(value)
        assert value > 0, '{} must be greater than 0'.format(dim)
        if value > 100:
            warnings.warn('{} is large, MDP may be slow'.format(dim))
        return value


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


class GridWorld(Domain, MDP):
    """ GridWorld domain

    A discrete world with cells (free, obstacles, and goal). The main task
    is to find a path from any start cell to a goal cell.

    """
    def __init__(self, gmap):
        gr = GridReward(domain=self)
        gt = GTransition(domain=self)

        # Domain.__init__(self)
        MDP.__init__(self, discount=0.9, reward=gr, transition=gt)

        gmap = np.asarray(gmap)
        assert gmap.ndim == 2, '`gmap` must be a two array'
        self._initialize(gmap)

    def _initialize(self, gmap):
        height, width = gmap.shape
        self._grid = Grid2D(nrows=width, ncols=height)

        self._terminals = list()
        self._states = dict()

        state_id = 0
        for i in range(self._grid.rows):
            for j in range(self._grid.cols):
                if gmap[i, j] == 1:
                    self._grid.block_cell((j, i))
                if gmap[i, j] == 2:
                    self._terminals.append((j, i))
                    self._goal = (j, i)

                self._states[state_id] = (j, i)
                state_id += 1

        self._action = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

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
        assert isinstance(state, Iterable), '`state` expected a tuple/list'
        return state in self._terminals

    def visualize(self, ax, **kwargs):
        if 'show_policy' in kwargs and 'policy' in kwargs:
            print('showing policy')

        cz = 1  # cell size
        for c in self._grid.cells:
            i, j = c[0], c[1]
            cell = (i * cz, j * cz)
            if self._grid.blocked((i, j)):
                ax.add_artist(Rectangle(cell, cz, cz, fc='r', ec='k'))
            elif self.terminal((i, j)):
                ax.add_artist(Rectangle(cell, cz, cz, fc='g', ec='k'))
            else:
                ax.add_artist(Rectangle(cell, cz, cz, fc='w', ec='k'))

        ax.set_xlim([0, self._grid.cols])
        ax.set_ylim([0, self._grid.rows])
        ax.set_xticks([])
        ax.set_yticks([])

        return ax

    def _show_policy(self, ax, **kwargs):
        pass
