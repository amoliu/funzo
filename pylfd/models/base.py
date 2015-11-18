
import six

from abc import ABCMeta
from abc import abstractmethod

import numpy as np


from ..base import Model


class MDPReward(six.with_metaclass(ABCMeta, Model)):
    """ Reward  function base class """

    _template = '_feature_'

    def __init__(self, world):
        # keep a reference to parent MDP to get access to domain and dynamics
        self._world = world

    @abstractmethod
    def __call__(self, state, action):
        """ Evaluate the reward function for the (state, action) pair

        Compute :math:`r(s, a) = \sum_i w_i f_i(s, a)`
        where :math:`f` is a (linear )function approximation for the reward
        parameterized by :math:`w`

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
        assert 0.0 <= discount < 1.0, 'The `discount` must be in [0, 1)'

        self.gamma = discount
        self.reward = reward  # keep a reference to reward function object

    @abstractmethod
    def terminal(self, state):
        """ Check if a state is terminal (goal state) """
        raise NotImplementedError('Abstract method')


###############################################################################


class Domain(six.with_metaclass(ABCMeta, Model)):

    """ Domain interface

    Domains are extensions of MDPs, have all mdp relevant information

    domain summarizes the following:

    MDP states and actions
    MDP dynamics/transitions

    MDP contains: discounting, terminal states?

    """

    def __init__(self, kind='discrete'):
        self.kind = kind

    @abstractmethod
    def visualize(self, ax, **kwargs):
        """ visualize a domain

        Parameters
        -----------
        ax : `matplotlib` axes
            Axes on which to visualize the domain
        kwargs : dict
            Optional key-world arguiments

        Returns
        --------
        ax : `matplotlib` axes
            The axis with the visual elements on the domain drawn

        """
        return NotImplementedError('This method is abstract')
