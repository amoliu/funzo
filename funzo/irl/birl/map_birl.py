"""
MAP-BIRL

Gradient based BIRL returning the MAP estimate of the reward distribution

"""

from __future__ import division

import logging

import numpy as np
from autograd import grad

import scipy as sp
from scipy.misc import logsumexp

from .base import BIRL
from ...utils.validation import check_random_state


logger = logging.getLogger(__name__)


class MAPBIRL(BIRL):
    """ MAP based BIRL """
    def __init__(self, mdp, prior, demos, planner, beta,
                 learning_rate=0.5, max_iter=50, verbose=4):
        super(MAPBIRL, self).__init__(mdp, prior, demos, planner, beta)
        # TODO - sanity checks
        self._learning_rate = learning_rate
        self._max_iter = max_iter

        # setup logger
        logging.basicConfig(level=verbose)

    def run(self, **kwargs):
        r = self._initialize_reward()

        rmax = self._mdp.reward.rmax
        bounds = tuple((-rmax, rmax)
                       for _ in range(len(self._mdp.reward)))

        # sum to 1 (or 1 - sum = 0)
        # only used with linear function approximation reward
        constraints = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})

        self._iter = 1
        # r is argmax_r p(D|r)p(r)
        res = sp.optimize.minimize(fun=self._reward_log_posterior,
                                   x0=r,
                                   # method='L-BFGS-B',
                                   method='SLSQP',
                                   jac=False,
                                   # jac=self._grad_llk,
                                   bounds=bounds,
                                   constraints=constraints,
                                   callback=self._callback_optimization)
        print(res)

        return res.x

    def _initialize_reward(self, random_state=0):
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
            H = len(traj)
            for (s, a) in traj:
                alpha_H = self._beta * Q_r[a, s]
                beta_Hs = list()
                for b in self._mdp.A:
                    beta_Hs.append(self._beta * Q_r[b, s])
                beta_H = logsumexp(beta_Hs)

            llk += (alpha_H - beta_H) / float(H+1)
        llk /= float(M)
        return llk

    def _grad_llk(self, r):
        """ Gradient of the reward log likelihood """
        return grad(self._reward_log_likelihood(r))

    def _callback_optimization(self, param):
        """ Callback to catch the optimization progress """
        logger.info('iter: {}, r: {}'.format(self._iter, param))

    def _reward_log_posterior(self, r):
        """ Compute the log posterior distribution of the current reward

        Compute :math:`\log p(\Xi | r) p(r)` with respect to the given
        reward

        """
        log_lk = self._reward_log_likelihood(r)
        log_prior = np.sum(self._prior.log_p(r))

        self._iter += 1

        return log_lk + log_prior
