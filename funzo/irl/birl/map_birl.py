"""
MAP-BIRL

Gradient based BIRL returning the MAP estimate of the reward distribution

"""

from __future__ import division

import logging

import scipy as sp
from scipy.misc import logsumexp

import numpy as np

from .base import BIRL


logger = logging.getLogger(__name__)


class MAPBIRL(BIRL):
    """ MAP based BIRL """
    def __init__(self, mdp, prior, demos, beta, eta, max_iter, verbose=4):
        super(MAPBIRL, self).__init__(mdp, prior, demos, beta)
        # TODO - sanity checks
        self._eta = eta
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

            # compute Q for current reward
            #
            # perform gradient step
            pass

        return r

    def _initialize_reward(self):
        d = self._mdp.reward.dim
        rmax = self._mdp.reward.rmax
        reward = np.array([np.random.uniform(-rmax, rmax) for _ in range(d)])
        return reward

    def _neg_loglk(self, r):
        """ Compute the negative log likelihood for r

        Compute :math:`\log p(\Xi | r) p(r)` with respect to the given
        reward

        """
        # - prepare the trajectory quality scores
        QE = self._rep.trajectory_quality(r, self._demos)
        QPi = [self._rep.trajectory_quality(r, self._g_trajs[i])
               for i in range(self._iteration)]

        # - the negative log likelihood
        # data term
        z = []
        for q_e in QE:
            for QP_i in QPi:
                for q_i in QP_i:
                    z.append(self._beta * (q_i - q_e))
        lk = -logsumexp(z)

        # prior term
        prior = np.sum(self._prior.log_p(r))

        return lk - prior
