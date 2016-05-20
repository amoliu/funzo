"""
Controllers (transition functions) for social navigation

"""

from __future__ import division

import numpy as np

from ..base import model_domain
from ..geometry import edist
from ...models import MDPLocalController

from .nav_world import SocialNavigationWorld


__all__ = ['LinearController']


class LinearController(MDPLocalController):
    """ Linear local controller

    Connect two pairs of states in :math:`\mathbb{R}^n` using a straight line
    with equally placed way-points as a specified resolution. The direction of
    the line is specified by the action which represents the continuous action.


    """
    def __init__(self, resolution=0.1, domain=None):
        super(LinearController, self).__init__(domain)
        self._domain = model_domain(domain, SocialNavigationWorld)

        if resolution < 1e-05:
            ValueError('Linear controller resolution is too low!')
        self._resolution = resolution

    def __call__(self, state, action, duration, **kwargs):
        """ Execute the local controller

        Run the local controller for a specified duration, and return the
        resulting trajectory.

        Parameters
        -----------
        state : int
            State in an MDP (usually represented using a controller graph)
        action : float
            Action to take (here corresponding to an angle in [0, 2pi])

        duration : float
            Time to run the controller. The exact advancement depends of the
            speed of the robot


        Returns
        --------
        traj : array-like
            A 2D trajectory with waypoints :math:`(x, y, \theta,
            v_{\text{max}})`
            Will return `None` is the controller drives the robot outside of
            the world.

        """
        state_ = self._domain.states[state]
        speed = kwargs.get('speed', 1.0)

        nx = state_[0] + np.cos(action) * duration
        ny = state_[1] + np.sin(action) * duration

        if self._world.in_world((nx, ny)):
            target = [nx, ny, action, speed]
            traj = self.trajectory(state, target, speed)
            return traj

        return None

    def trajectory(self, source, target, **kwargs):
        """ Compute the local trajectory to connect two states """
        source = np.asarray(source)
        target = np.asarray(target)
        V = kwargs.get('speed', 1.0)

        duration = edist(source, target)
        dt = (V * duration) * 1.0 / self._resolution
        theta = np.arctan2(target[1] - source[1], target[0] - source[0])

        traj = [target[0:2] * t / dt + source[0:2] * (1 - t / dt)
                for t in range(int(dt))]
        traj = [t.tolist() + [theta, V] for t in traj]
        traj = np.array(traj)
        return traj
