"""
GridWorld Domain
"""

from __future__ import division

import six
import numpy as np

from ..base import Model
from ..models.base import Domain


class Grid(Model):
    """
    Represents a rectangular grid map. The map consists of
    nrows X ncols coordinates (squares). Some of the squares
    can be blocked (by obstacles).

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
        """ Print the map to stdout in ASCII
        """
        for row in range(self._rows):
            for col in range(self._cols):
                print("%s" % ('[X]' if self._map[row, col] == 1 else '[ ]')),
            print('')

    def __str__(self):
        return self.__repr__

    def _check_dim(self, value, dim):
        assert value > 0, '{} must be greater than 0'.format(dim)
        assert isinstance(value, int), '{} must be an integer'.format(dim)
        if value > 100:
            warnings.warn('{} is large, MDP may be slow'.format(dim))
        return value


class GridWorld(Domain):
    """ GridWorld domain

    A discrete world with cells (free, obstacles, and goal). The main task
    is to find a path from any start cell to a goal cell.

    """
    def __init__(self, gmap, goal):
        super(GridWorld, self).__init__(kind='discrete')
        self._gmap = gmap
        self._goal = goal

    def visualize(self, ax, **kwargs):
        if 'show_policy' in kwargs and 'policy' in kwargs:
            print('showing policy')

        pass
