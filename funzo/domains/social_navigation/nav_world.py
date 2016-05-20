
from __future__ import division

import warnings

import numpy as np

from matplotlib.patches import Circle, Ellipse


__all__ = ['SocialNavigationWorld']


class SocialNavigationWorld(object):
    """ Social navigation domain

    A mobile robot is to move in a crowded environment. The scene is populated
    with people, some of whom have pair-wise relations and additional semantic
    attributes. Additional semantic entities beyond simple obstacles can also
    be placed in the environment. The task is to navigate while respecting the
    social constrains which are defined based on these high level attributes,
    e.g. being polite may mean not crossing any pair-wise relations

    The environment is modeled as a bounded rectangle, :math:`(x, y, w, h)` and
    entities are grounded on this representation. For example people are given
    as an array of :math:`(x_p, y_p, \theta_p)`. The goal position is a 2D
    location.

    .. note:: The environment is continuous which is one of the main challenges


    Parameters
    -----------
    x, y, w, h : float
        Start coordinates (x, y) and environment extents in 2D (w, h)
    goal : array-like
        2D location of the goal position in the world
    entities : dict
        Key-value pairs of additional semantic elements such as:
            * People `persons` as a dict of :math:`(x, y, \theta)`
            * Person pair-wise groupings `groups` as a 2D array of person
            ids.

    Attributes
    ------------
    x, y, w, h : float
        Start coordinates (x, y) and environment extents in 2D (w, h)
    goal : array-like
        2D location of the goal position in the world
    _persons : dict
        2D person poses indexed by id
    _groups : array-like
        2D array of person ids indicating pair-wise relationships

    """

    def __init__(self, x, y, w, h, goal, **entities):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        if not self.in_domain(goal):
            raise ValueError('Goal location outside of world limits')
        self.goal = goal

        self._persons = entities.get('persons', None)
        self._groups = entities.get('groups', None)

    def in_domain(self, state):
        """ Check is the state is within the social navigation world limits"""
        return self.x < state[0] < self.w and\
            self.y < state[1] < self.h

    def terminal(self, state):
        """ Check if a state is the goal state """
        state_ = self.states[state]
        return np.linalg.norm(state_, self.goal) < 1e-02

    def visualize(self, ax, **kwargs):
        """ Visualize the social navigation scene """
        if self._persons is None:
            warnings.warn('No entities to visualize!')
            return ax

        for _, p in self._persons.items():
            phead = np.degrees(np.arctan2(p[3], p[2]))
            ax.add_artist(Ellipse((p[0], p[1]), width=0.3, height=0.6,
                                  angle=phead, color='r', fill=False,
                                  lw=1.5, aa=True, zorder=3))
            ax.add_artist(Circle((p[0], p[1]), radius=0.12, color='w',
                                 ec='r', lw=2.5, aa=True, zorder=3))
            ax.arrow(p[0], p[1], p[2] / 5., p[3] / 5., fc='r', ec='r',
                     lw=1.5, head_width=0.14, head_length=0.1, zorder=3)

        if self._groups is not None:
            for (i, j) in self._groups:
                x1, y1 = self._persons[i][0], self._persons[i][1]
                x2, y2 = self._persons[j][0], self._persons[j][1]
                ax.plot((x1, x2), (y1, y2), ls='-', c='r', lw=2.0, zorder=2)

        return ax
