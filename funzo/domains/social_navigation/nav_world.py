
import numpy as np


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


    """

    def __init__(self, x, y, w, h, goal, **entities):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        if not self.in_domain(goal):
            raise ValueError('Goal location outside of world limits')
        self.goal = goal

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
        return ax
