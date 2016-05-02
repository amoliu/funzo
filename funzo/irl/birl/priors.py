"""
Reward function prior distributions

"""

from __future__ import division

import six

import scipy.stats
import numpy as np

from abc import abstractmethod, ABCMeta

from ..base import Model


__all__ = [
    'RewardPriorBase',
    'UniformRewardPrior',
    'GaussianRewardPrior',
]


class RewardPriorBase(six.with_metaclass(ABCMeta, Model)):
    """ Reward prior distribution API

    The reward prior summarizes information about the reward distribution that
    is available before running the algorithm, i.e. all the relevant domain
    knowledge.

    .. note:: These distributions are multivariate, i.e. reward samples are
        vectors or equivalently functions over :math:`\mathcal{S}`, or more
        generally :math:`\mathcal{S} \\times \mathcal{A}` or subsets of these.

    """

    def __init__(self, dim):
        if 0 > dim:
            raise ValueError('Reward space dimension must be positive')
        self._dim = dim

    @abstractmethod
    def pdf(self, r):
        """ Estimate the probability of the reward under the prior

        .. math::

            p(r \in A) = \int_A f d\mu

        for any :math:`A \in \mathcal{A}`, given some measurable space
        :math:`(\mathcal{X}, \mathcal{A})` and a measure :math:`\mu`.

        """
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def log_p(self, r):
        """ Estimate the log probability of the reward under the prior """
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def sample(self):
        """ Generate a sample from the reward prior distribution

        .. math::

            r \sim f_{\\theta}

        """
        raise NotImplementedError('Abstract method')


class UniformRewardPrior(RewardPriorBase):
    """ Uniform reward prior distribution

    Suitable to task in which there is no clear insight into the nature of the
    reward function.

    .. math:: p(r(s, a) = x) = \\text{Uni}(a, b)

    """
    def __init__(self, dim=1, rmin=0.0, rmax=1.0):
        super(UniformRewardPrior, self).__init__(dim)
        if rmax < rmin:
            raise ValueError('Dist rmax cannot be less than rmin')
        self._dist = scipy.stats.uniform(loc=rmin, scale=2 * (rmax - rmin))

    def pdf(self, r):
        """ Estimate the probability of the reward under the prior """
        return np.prod([self._dist.pdf(x) for x in r])

    def log_p(self, r):
        """ Estimate the log probability of the reward under the prior """
        return np.sum(self._dist.logpdf(x) for x in r)

    def sample(self):
        """ Generate a sample from the reward prior distribution """
        return self._dist.rvs(size=self._dim)


class GaussianRewardPrior(RewardPriorBase):
    """ Gaussian reward prior distribution

    Suitable for many real world tasks with parsimonious reward structures,
    where most states have negligible rewards [RamBIRL07]_.

    .. math:: p(r(s, a) = x) = \\frac{1}{\sqrt{2\pi}\sigma}
        \exp\left(-\\frac{x^2}{2\sigma^2}\\right)

    .. [RamBIRL07] Deepak Ramachandran and Eyal Amir, "Bayesian inverse
        reinforcement learning," IJCAI, 2007

    """
    def __init__(self, dim=1, mean=0.0, sigma=0.5):
        super(GaussianRewardPrior, self).__init__(dim)
        self._dist = scipy.stats.norm(loc=mean, scale=sigma)

    def pdf(self, r):
        """ Estimate the probability of the reward under the prior """
        return np.prod([self._dist.pdf(x) for x in r])

    def log_p(self, r):
        """ Estimate the log probability of the reward under the prior """
        return np.sum(self._dist.logpdf(x) for x in r)

    def sample(self):
        """ Generate a sample from the reward prior distribution

        .. math::

            r \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\sigma})

        """
        return self._dist.rvs(size=self._dim)
