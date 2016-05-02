"""
BIRL algorithms with Optimization based inference. Includes;
    - MAP
    - Gradient Descent based approaches
    -
"""

from __future__ import division, absolute_import

import numpy as np

from six.moves import range
from scipy.optimize import minimize

from .birl_base import BIRLBase
from ...utils.data_structures import Trace


__all__ = ['MAPBIRL']


class MAPBIRL(BIRLBase):
    """ BIRL using MAP for inference

    .. note:: Returns a single estimate of the reward function

    """
    def __init__(self, prior, beta=0.7, max_iter=100,
                 planner=None, random_state=None):
        super(MAPBIRL, self).__init__(prior, beta, planner, random_state)

        if 0 >= max_iter > np.inf:
            raise ValueError('No. of iterations must be in (0, inf)')
        self._max_iter = max_iter

    def solve(self, demos, mdp=None):
        """ Solve the BIRL problem using MAP """
        if mdp is None:
            raise ValueError('BIRL requires an MDP model')

        v = ['r', 'f', 'r_map']
        trace = Trace(v, save_interval=0)

        rmax = mdp.reward.rmax
        bounds = tuple((-rmax, rmax) for _ in range(len(mdp.reward)))

        def _callback_optimization(x):
            """ Callback to catch the optimization progress """
            trace.record(r=x)

        def _objective(r):
            """ Objective function """
            plan_r = self.solve_mdp(mdp, r)
            return -self.log_posterior(r, demos, mdp, plan_r)

        res = minimize(fun=_objective,
                       x0=self.initialize_reward(),
                       method='L-BFGS-B',
                       jac=False,
                       bounds=bounds,
                       callback=_callback_optimization)

        trace.record(r_map=res.x, f=res.fun)

        return trace
