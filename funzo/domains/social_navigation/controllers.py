"""
Controllers (transition functions) for social navigation

"""

from __future__ import division

import numpy as np

from ..base import model_domain
from ..geometry import edist, normangle
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
            ValueError('Local controller resolution is too low!')
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
            A 2D trajectory with way-points :math:`(x, y, \theta,
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
            traj = self.trajectory(state, target, speed=speed)
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
        # traj = [t.tolist() + [theta, V] for t in traj]
        traj = [t.tolist() + [theta] for t in traj]
        traj = np.array(traj)
        return traj


class POSQController(LinearController):
    """ POSQ local controller for differential drive mobile robots

    A two-point boundary value problem solver with guaranteed convergence. For
    a detailed exposition, see [Palmieri14]_.

    .. [Palmieri14] Luigi Palmieri and Kai O. Arras, "A Novel RRT Extend
    Function for Efficient and Smooth Mobile Robot Motion Planning," IROS,
    2014

    """
    def __init__(self, baseline=0.4, direction=1, resolution=0.1, domain=None):
        super(POSQController, self).__init__(resolution, domain)
        self._domain = model_domain(domain, SocialNavigationWorld)
        if baseline < 0.0:
            raise ValueError('Diff-drive robot baseline must be > 0.0')
        self._base = baseline
        if direction not in [-1, 1]:
            raise ValueError('POSQ direction can only be -1 or 1')
        self._direction = direction

    def trajectory(self, source, target, **kwargs):
        """ Compute the local trajectory to connect two states """
        theta = np.arctan2(target[1] - source[1], target[0] - source[0])
        source = np.array([source[0], source[1], theta])
        target = np.array([target[0], target[1], theta])
        V = kwargs.get('speed', 1.0)

        init_t = 0
        traj, speedvec, vel, inct = self._posq_integrate(
            source, target, self._direction, self._resolution,
            self._base, init_t, V, nS=0)

        # speeds = np.hypot(speedvec[:, 0], speedvec[:, 1])
        # traj = np.column_stack((traj, speeds))

        return traj

    def _posq_integrate(self, xstart, xend, direction, deltaT,
                        base, initT, vmax, nS=0):
        """ POSQ Integration procedure to general full trajectory """
        if xstart.shape != xend.shape:
            raise ValueError('Expect similar vector sizes in POSQ integrate')

        vel = np.zeros(shape=(1, 2))
        sl, sr = 0, 0
        old_sl, old_sr = 0, 0
        xvec = np.zeros(shape=(1, 3))  # pose vectors for trajectory
        speedvec = np.zeros(shape=(1, 2))  # velocities during trajectory
        encoders = [0, 0]
        t = initT  # initialize global timer
        ti = 0  # initialize local timer
        eot = 0  # initialize end-of-trajectory flag
        xnow = [xstart[0], xstart[1], xstart[2]]
        old_beta = 0

        while not eot:
            # Calculate distances for both wheels
            dSl = sl - old_sl
            dSr = sr - old_sr
            dSm = (dSl + dSr) / 2.0
            dSd = (dSr - dSl) / base

            # Integrate robot position
            xnow[0] = xnow[0] + dSm * np.cos(xnow[2] + dSd / 2.0)
            xnow[1] = xnow[1] + dSm * np.sin(xnow[2] + dSd / 2.0)
            xnow[2] = normangle(xnow[2] + dSd, -np.pi)

            # implementation of the controller
            vl, vr, eot, vm, vd, old_beta = self._posq_step(ti, xnow, xend,
                                                            direction,
                                                            old_beta, vmax)
            vel = np.row_stack((vel, [vm, vd]))
            speeds = np.array([vl, vr])
            speedvec = np.row_stack((speedvec, speeds))
            xvec = np.row_stack((xvec, xnow))

            # Increase timers
            ti = ti + deltaT
            t = t + deltaT

            # Increase accumulated encoder values
            # simulated encoders of robot
            delta_dist1 = speeds[0] * deltaT
            delta_dist2 = speeds[1] * deltaT
            encoders[0] += delta_dist1
            encoders[1] += delta_dist2

            # Keep track of previous wheel positions
            old_sl = sl
            old_sr = sr

            # noise on the encoders
            sl = encoders[0] + nS * np.random.uniform(0, 1)
            sr = encoders[1] + nS * np.random.uniform(0, 1)

        inct = t  # at the end of the trajectory the time elapsed is added

        return xvec, speedvec, vel, inct

    def _posq_step(self, t, xnow, xend, direction, old_beta, vmax):
        """ POSQ single step """
        k_v = 3.8
        k_rho = 1    # Condition: k_alpha + 5/3*k_beta - 2/pi*k_rho > 0 !
        k_alpha = 6
        k_beta = -1
        rho_end = 0.00510      # [m]

        if t == 0:
            old_beta = 0

        # extract coordinates
        xc, yc, tc = xnow[0], xnow[1], xnow[2]
        xe, ye, te = xend[0], xend[1], xend[2]

        # rho
        dx = xe - xc
        dy = ye - yc
        rho = np.sqrt(dx**2 + dy**2)
        f_rho = rho
        if f_rho > (vmax / k_rho):
            f_rho = vmax / k_rho

        # alpha
        alpha = normangle(np.arctan2(dy, dx) - tc, -np.pi)

        # direction (forward or backward)
        if direction == 1:
            if alpha > np.pi / 2:
                f_rho = -f_rho                   # backwards
                alpha = alpha - np.pi
            elif alpha <= -np.pi / 2:
                f_rho = -f_rho                   # backwards
                alpha = alpha + np.pi
        elif direction == -1:                  # arrive backwards
            f_rho = -f_rho
            alpha = alpha + np.pi
            if alpha > np.pi:
                alpha = alpha - 2 * np.pi

        # phi, beta
        phi = te - tc
        phi = normangle(phi, -np.pi)
        beta = normangle(phi - alpha, -np.pi)
        if abs(old_beta - beta) > np.pi:           # avoid instability
            beta = old_beta
        old_beta = beta

        vm = k_rho * np.tanh(f_rho * k_v)
        vd = (k_alpha * alpha + k_beta * beta)
        eot = (rho < rho_end)

        # Convert speed to wheel speeds
        vl = vm - vd * self._base / 2
        if abs(vl) > vmax:
            vl = vmax * np.sign(vl)

        vr = vm + vd * self._base / 2
        if abs(vr) > vmax:
            vr = vmax * np.sign(vr)

        return vl, vr, eot, vm, vd, old_beta
