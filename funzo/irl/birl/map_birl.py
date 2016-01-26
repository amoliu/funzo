"""
MAP-BIRL

Gradient based BIRL returning the MAP estimate of the reward distribution

"""

from __future__ import division

import logging

import numpy as np
from autograd import grad
from scipy.misc import logsumexp

from .base import BIRL
from ...utils.validation import check_random_state


logger = logging.getLogger(__name__)


class MAPBIRL(BIRL):
    """ MAP based BIRL """
    def __init__(self, mdp, prior, demos, beta,
                 learning_rate, max_iter, verbose=4):
        super(MAPBIRL, self).__init__(mdp, prior, demos, beta)
        # TODO - sanity checks
        self._learning_rate = learning_rate
        self._max_iter = max_iter

        # setup logger
        logging.basicConfig(level=verbose)

    def run(self, **kwargs):
        r = self._initialize_reward()
        self._mdp.reward.weights = r
        plan = self._planner(self._mdp)
        logger.info(plan['Q'])

        for step in range(1, self._max_iter + 1):
            # update reward
            r = self._eta * r

            # posterior for the current reward
            posterior = self._reward_posterior(r, plan['Q'])

            # perform gradient step
            r = self._eta * posterior * r

        return r

    def _initialize_reward(self, random_state=0):
        """ Initialize a reward vector using the prior """
        rng = check_random_state(random_state)
        r = rng.rand(self._mdp.reward.dim)
        return self._prior(r)

    def _reward_log_likelihood(self, r):
        """ Compute the reward log likelihood using the new reward and data

        i.e. :math:`p(\Xi | r) = ...`

        """
        self._mdp.reward.weights = r
        plan = self._planner(self._mdp)
        Q_r = plan['Q']

        data_llk = 0.0
        for traj in self._demos:
            for (s, a) in traj:
                den = logsumexp(self._beta * Q_r[s, b] for b in self._mdp.A)
                data_llk += (self._beta * Q_r[s, a]) - den

        return data_llk

    def _grad_llk(self, r):
        """ Gradient of the reward log likelihood """
        return grad(self._reward_log_likelihood(r))

    def _reward_log_posterior(self, r):
        """ Compute the log posterior distribution of the current reward

        Compute :math:`\log p(\Xi | r) p(r)` with respect to the given
        reward

        """
        log_lk = self._reward_log_likelihood(r)
        log_prior = np.sum(self._prior.log_p(r))

        return log_lk + log_prior
