"""
Bayesian (type) inverse reinforcement learning

"""
from abc import ABCMeta, abstractmethod

import six

import numpy as np

from scipy.stats import uniform, norm, laplace

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
    planner : a callable
        A planner for MDP e.g. policy iteration as a callable
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
    _planner : a callable
        A reference to a planner for MDP e.g. policy iteration as a callable
    _beta : float, optional (default=0.9)
        Expert optimality parameter for the reward likelihood term in the
        product of exponential distributions

    """

    def __init__(self, mdp, prior, demos, planner, beta=0.7):
        self._mdp = mdp
        self._prior = prior
        self._planner = planner
        self._demos = demos
        self._beta = beta

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

    def __init__(self, loc=0.0, scale=1.0):
        self._loc = loc
        self._scale = scale

    def __call__(self, r):
        return uniform.pdf(r, loc=self._loc, scale=self._scale)

    def log_p(self, r):
        return uniform.logpdf(r, loc=self._loc, scale=self._scale)


class GaussianRewardPrior(RewardPrior):
    """Gaussian reward prior"""
    def __init__(self, sigma=0.5, loc=0.0):
        self._sigma = sigma
        self._loc = loc

    def __call__(self, r):
        return norm.pdf(r, loc=self._loc, scale=self._sigma)

    def log_p(self, r):
        return norm.logpdf(r, loc=self._loc, scale=self._sigma)


class LaplacianRewardPrior(RewardPrior):
    """Laplacian reward prior"""
    def __init__(self, sigma=0.5, loc=0.0):
        self._sigma = sigma
        self._loc = loc

    def __call__(self, r):
        return laplace.pdf(r, loc=self._loc, scale=self._sigma)

    def log_p(self, r):
        return laplace.logpdf(r, loc=self._loc, scale=self._sigma)


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
        return rp

    def log_p(self, r):
        return np.log(self.__call__(r))


########################################################################
# Common utilities
# ######################################################################

class Trace(object):
    """ MCMC sampling trace """
    def __init__(self, variables):
        self.data = dict()
        self.data['r'] = None
        self.data['step'] = list()
        self.data['sample'] = list()
        self.data['accept'] = list()
        self.data['Q_r'] = list()

    def record(self, r, step, sample, accept, Q):
        self.data['r'] = r
        self.data['step'].append(step)
        self.data['sample'].append(sample)
        self.data['accept'].append(accept)
        self.data['Q_r'].append(Q)

    def save(self, filename='trace'):
        """ Save as HDF5 """
        # get the time and append to name
        pass

    def plot(self, axes):
        pass
