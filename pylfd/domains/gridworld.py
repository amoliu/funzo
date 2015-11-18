"""
GridWorld Domain
"""

from __future__ import division

from collections import Iterable
from matplotlib.patches import Rectangle

import six
import numpy as np

from ..base import Model
from ..models.base import Domain, MDP, MDPReward


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
    def __init__(self, world):
        super(GridReward, self).__init__(world)

    def __call__(self, state, action):
        return 1.0

    @property
    def dim():
        return len(self._world.S)


class GridWorld(Domain, MDP):
    """ GridWorld domain

    A discrete world with cells (free, obstacles, and goal). The main task
    is to find a path from any start cell to a goal cell.

    """
    def __init__(self, gmap):
        g_reward = GridReward(world=self)

        Domain.__init__(self, kind='discrete')
        MDP.__init__(self, discount=0.9, reward=g_reward)

        gmap = np.asarray(gmap)
        assert gmap.ndim == 2, '`gmap` must be a two array'
        self._initialize(gmap)

    def _initialize(self, gmap):
        height, width = gmap.shape
        self._grid = Grid2D(nrows=width, ncols=height)

        self.terminals = list()
        self.S = dict()

        print(self._grid.rows, self._grid.cols)

        state_id = 0
        self.state_map = dict()
        for i in range(self._grid.rows):
            for j in range(self._grid.cols):
                if gmap[i, j] == 1:
                    self._grid.block_cell((j, i))
                if gmap[i, j] == 2:
                    self.terminals.append((j, i))
                    self._goal = (j, i)

                self.S[state_id] = (j, i)
                self.state_map[(j, i)] = state_id
                state_id += 1

        self.A = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

    def transition(self, state_id, action_id):
        """ Transition model.  From a state and an action, return a list
        of (probability, result-state) pairs.
        """
        state = self.S[state_id]
        action = self.A[action_id]
        if action is None:
            return [(0.0, state)]
        else:
            return [(0.8, self._move(state, action)),
                    (0.1, self._move(state, _right(action, self.A.values()))),
                    (0.1, self._move(state, _left(action, self.A.values())))]

    def actions(self, state):
        """ Set of actions that can be performed in this state. """
        if self.terminal(self.S[state]):
            return self.A.keys()

    def terminal(self, state):
        """ Check if a state is terminal"""
        assert isinstance(state, Iterable), '`state` expected a tuple/list'
        return state in self.terminals

    def _move(self, state, direction):
        """ Return the state that results from going in this direction. Stays
        in the same state if action os leading to go outside the world or to
        obstacles
        """
        ns = (state[0]+direction[0], state[1]+direction[1])
        if not self._grid.valid_cell(ns):
            return self.state_map[state]
        if self._grid.blocked(ns):
            return self.state_map[state]
        return self.state_map[ns]

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


#############################################################################
# Grid world controller

def _heading(heading, inc, directions):
    return directions[(directions.index(heading) + inc) % len(directions)]


def _right(heading, directions):
    return _heading(heading, -1, directions)


def _left(heading, directions):
    return _heading(heading, +1, directions)


# Create a class for gridworl controller and reward function
#
