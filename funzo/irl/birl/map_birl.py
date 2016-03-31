"""
MAP-BIRL

BIRL using the MAP the reward posterior distribution as estimated reward
function.

"""

from __future__ import division, absolute_import

import logging
import warnings

import numpy as np

from scipy.optimize import minimize
from scipy.misc import logsumexp

from .base import BIRL

from ...utils.data_structures import Trace


logger = logging.getLogger(__name__)


class MAPBIRL(BIRL):
    """ MAP based BIRL """
    def __init__(self, mdp, prior, demos, planner, beta,
                 max_iter=50, verbose=4):
        super(MAPBIRL, self).__init__(mdp, prior, demos, planner, beta)
        if 0 > max_iter:
            raise ValueError('*max_iter* cannot be negative')
        if max_iter > 5000:
            warnings.warn('Large number of iterations my take long times')
        self._max_iter = max_iter

        logging.basicConfig(level=verbose)

    def run(self, **kwargs):
        """ Run the algorithm with the specified parameters """
        rseed = kwargs.get('random_state', None)
        r = self._initialize_reward(random_state=rseed)
        self._iter = 0

        self._trace = Trace(variables=['r', 'r_map', 'step', 'f'],
                            save_interval=self._max_iter//2)
        self._trace.record(r=r, step=self._iter)

        rmax = self._mdp.reward.rmax
        bounds = tuple((-rmax, rmax) for _ in range(len(self._mdp.reward)))

        # r is argmax_r p(D|r)p(r)
        # Sequential Least SQuares Programming (SLSQP)
        res = minimize(fun=self._log_posterior,
                       x0=r,
                       method='L-BFGS-B',
                       jac=False,
                       bounds=bounds,
                       callback=self._callback_optimization)

        self._trace.record(r_map=res.x, f=res.fun)

        logger.info('Termination: {}, Iters: {}'.format(res.success, res.nit))

        return self._trace

    def _initialize_reward(self, random_state=None):
        """ Initialize a reward vector using the prior """
        return self._prior.sample(dim=len(self._mdp.reward))

    def _log_likelihood(self, r):
        """ Compute the reward log likelihood using the new reward and data

        i.e. :math:`p(\Xi | r) = ...`

        """
        self._mdp.reward.weights = r
        Q_r = self._planner(self._mdp)['Q']

        M = len(self._demos)
        llk = 0.0
        for traj in self._demos:
            if traj:
                H = len(traj)
                alpha_H = 0.0
                beta_H = 0.0
                for (s, a) in traj:
                    alpha_H += self._beta * Q_r[a, s]
                    beta_Hs = list()
                    for b in self._mdp.A:
                        beta_Hs.append(self._beta * Q_r[b, s])
                    beta_H += logsumexp(beta_Hs)

                llk += (alpha_H - beta_H) / float(H)
        llk /= float(M)
        return llk

    def _callback_optimization(self, x):
        """ Callback to catch the optimization progress """
        self._iter += 1

        self._trace.record(r=x, step=self._iter)
        logger.info('iter: {}, Reward: {}'.format(self._iter, x))

    def _log_posterior(self, r):
        """ Compute the negative log posterior distribution

        Compute :math:`-\log p(\Xi | r) p(r)` with respect to the given
        reward

        """
        log_lk = self._log_likelihood(r)
        log_prior = np.sum(self._prior.log_p(r))

        # -log p(r|D) = - log p(D|r) - log p(r)
        return -log_lk - log_prior
