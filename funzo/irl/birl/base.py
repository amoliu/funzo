"""
Bayesian (type) inverse reinforcement learning

"""
from abc import ABCMeta, abstractmethod

import six
import numpy as np

from ...base import Model


class BIRL(six.with_metaclass(ABCMeta, Model)):
    """ Base interface for BIRL algorithms

    BIRL algorithm that seeks to find a reward function underlying a set of
    expert demonstrations by computing the posterior of the reward distribution
    :math:`p(r | \Xi)`.

    These algorithms typically summarize the distribution by taking a single
    value such as the mean.

    Parameters
    ----------
    mdp : :class:`MDP` object or derivative
        The underlying MDP whose reward is sort
    prior : :class:`RewardPrior` or derivative object
        Reward prior callable object
    demos : array-like
        Expert demonstrations as set of :math:`M` trajectories of state action
        pairs. Trajectories can be of different lengths.
    beta : float, optional (default=0.7)
        Expert optimality parameter for the reward likelihood term in the
        product of exponential distributions

    Attributes
    ----------
    _mdp : :class:`MDP` object or derivative
        Reference to the underlying MDP whose reward is sort
    _prior : :class:`RewardPrior` or derivative object
        Reference to the reward prior callable object
    _demos : array-like
        Expert demonstrations as set of :math:`M` trajectories of state action
        pairs. Trajectories can be of different lengths.
    _beta : float, optional (default=0.9)
        Expert optimality parameter for the reward likelihood term in the
        product of exponential distributions

    """

    def __init__(self, mdp, prior, demos, beta=0.7):
        self._mdp = mdp
        self._prior = prior

    @abstractmethod
    def run(self, **kwargs):
        """ Run the BIRL algorithm to find a reward function """
        raise NotImplementedError('Abstract')


########################################################################
# Reward Priors
# ######################################################################


class RewardPrior(six.with_metaclass(ABCMeta, Model)):
    """ Reward prior interface

    The reward prior summarizes information about the reward distribution that
    is available before running the algorithm, i.e. all the relavant domain
    knowledge.

    """

    @abstractmethod
    def __call__(self, r):
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def log_p(self, r):
        raise NotImplementedError('Abstract method')


class UniformRewardPrior(RewardPrior):
    """ Uniform/flat prior"""

    def __call__(self, r):
        rp = np.ones(r.shape[0])
        dist = rp / np.sum(rp)
        return dist

    def log_p(self, r):
        return np.log(self.__call__(r))


class GaussianRewardPrior(RewardPrior):
    """Gaussian reward prior"""
    def __init__(self, sigma=0.5):
        self._sigma = sigma

    def __call__(self, r):
        rp = np.exp(-np.square(r)/(2.0*self._sigma**2)) /\
            np.sqrt(2.0*np.pi)*self._sigma
        return rp / np.sum(rp)

    def log_p(self, r):
        # TODO - make analytical
        return np.log(self.__call__(r))


class LaplacianRewardPrior(RewardPrior):
    """Laplacian reward prior"""
    def __init__(self, sigma=0.5):
        self._sigma = sigma

    def __call__(self, r):
        rp = np.exp(-np.fabs(r)/(2.0*self._sigma)) / (2.0*self._sigma)
        return rp / np.sum(rp)

    def log_p(self, r):
        # TODO - make analytical
        return np.log(self.__call__(r))


class DirectionalRewardPrior(RewardPrior):
    """ Prior that injects direction information

    Useful for cases in which we know the direction of influence of a feature
    as either a penalty or a reward. Defaults to all rewarding features

    """
    def __init__(self, directions=None):
        assert directions is not None, 'Directions must be a valid array'
        self.directions = directions

    def __call__(self, r):
        dim = len(r)
        rp = np.array([r[i] * self.directions[i] for i in range(dim)])
        return rp / np.sum(rp)

    def log_p(self, r):
        return np.log(self.__call__(r))
