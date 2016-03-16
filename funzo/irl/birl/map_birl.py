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
from ...utils.validation import check_random_state


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

        self._data = dict()
        self._data['rewards'] = list()
        self._data['iter'] = list()

    def run(self, **kwargs):
        """ Run the algorithm with the specified parameters """
        rseed = kwargs.get('random_state', None)
        r = self._initialize_reward(random_state=rseed)

        self._iter = 0

        self._data['rewards'].append(r)
        self._data['iter'].append(self._iter)

        rmax = self._mdp.reward.rmax
        bounds = tuple((-rmax, rmax) for _ in range(len(self._mdp.reward)))
        constraints = None

        # ensure weights sum to 1 (or 1 - sum = 0)
        # only used with linear function approximation reward
        if self._mdp.reward.kind == 'LFA':
            constraints = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})

        # r is argmax_r p(D|r)p(r)
        # Sequential Least SQuares Programming (SLSQP)
        res = minimize(fun=self._reward_log_posterior,
                       x0=r,
                       method='SLSQP',
                       jac=False,
                       bounds=bounds,
                       constraints=constraints,
                       callback=self._callback_optimization)

        logger.info('Termination: {}, Iters: {}'.format(res.success, res.nit))

        return res.x, self._data

    def _initialize_reward(self, random_state=None):
        """ Initialize a reward vector using the prior """
        rng = check_random_state(random_state)
        r = rng.rand(len(self._mdp.reward))
        return self._prior(r)

    def _reward_log_likelihood(self, r):
        """ Compute the reward log likelihood using the new reward and data

        i.e. :math:`p(\Xi | r) = ...`

        """
        self._mdp.reward.weights = r
        plan = self._planner(self._mdp)
        Q_r = plan['Q']

        M = len(self._demos)
        llk = 0.0
        for traj in self._demos:
            if traj:
                H = len(traj)
                alpha_H = 0.0
                for (s, a) in traj:
                    alpha_H += self._beta * Q_r[a, s]
                    beta_Hs = list()
                    for b in self._mdp.A:
                        beta_Hs.append(self._beta * Q_r[b, s])
                    beta_H = logsumexp(beta_Hs)

                llk += (alpha_H - beta_H) / float(H+1)
        llk /= float(M)
        return llk

    def _callback_optimization(self, x):
        """ Callback to catch the optimization progress """
        self._data['rewards'].append(x)
        self._data['iter'].append(self._iter)
        logger.info('iter: {}, Reward: {}'.format(self._iter, x))
        self._iter += 1

    def _reward_log_posterior(self, r):
        """ Compute the negative log posterior distribution

        Compute :math:`-\log p(\Xi | r) p(r)` with respect to the given
        reward

        """
        log_lk = self._reward_log_likelihood(r)
        log_prior = np.sum(self._prior.log_p(r))

        # -log p(r|D) = - log p(D|r) - log p(r)
        return -log_lk - log_prior
