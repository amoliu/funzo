
import six

from abc import ABCMeta
from abc import abstractmethod

import numpy as np


from ..base import Model


class MDPReward(six.with_metaclass(ABCMeta, Model)):
    """ Reward  function base class """

    _template = '_feature_'

    def __init__(self, world, kind='linfa'):
        # keep a reference to parent MDP to get access to S, A
        self._world = world
        self.kind = kind

    @abstractmethod
    def __call__(self, state, action):
        """ Evaluate the reward function for the (state, action) pair

        Compute :math:`r(s, a) = f(s, a, w)` where :math:`f` is a function
        approximation for the reward parameterized by :math:`w`
        """
        raise NotImplementedError('Abstract method')

    @property
    def dim(self):
        """ Dimension of the reward function """
        # - count all class members named '_feature_{x}'
        features = self.__class__.__dict__
        dim = sum([f[0].startswith(self._template) for f in features])
        return dim


class RewardLoss(six.with_metaclass(ABCMeta, Model)):
    """ Reward loss function """

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, r1, r2):
        """ Reward loss between ``r1`` and ``r2`` """
        raise NotImplementedError('Abstract')


class MDP(Model):
    """ Markov Decision Process Model

    Parameters
    ------------
    discount : float
        MDP discount factor
    reward : :class:`MDPReward` object
        Reward function for social navigation task

    Attributes
    -----------
    gamma : float
        MDP discount factor
    _reward : :class:`MDPReward` object
        Reward function for social navigation task

    """

    def __init__(self, discount, reward):
        if 0.0 > discount >= 1.0:
            raise ValueError('The `discount` must be in [0, 1)')

        self.gamma = discount
        self.reward = reward

    @abstractmethod
    def terminal(self, state):
        """ Check if a state is terminal (goal state) """
        raise NotImplementedError('Abstract method')
