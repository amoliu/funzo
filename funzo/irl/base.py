"""
Inverse reinforcement learning algorithms base modules

"""
from abc import ABCMeta, abstractmethod

import six
import numpy as np

from ..base import Model


__all__ = [
    'PolicyLoss',
    'RewardLoss',
]


########################################################################
# Loss functions
# ######################################################################

class Loss(six.with_metaclass(ABCMeta, Model)):
    """ A loss function for evaluating progress of IRL algorithms """

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, r_e, r_pi, **kwargs):
        # TODO - May enforce impmentation with ufuncs ?
        pass


class PolicyLoss(Loss):
    """ Policy loss with respect to a reward function

    L_p = || V^*(r) - V^{\pi}(r) ||_p

    """
    def __init__(self, mdp, planner, order=2):
        super(PolicyLoss, self).__init__(name='policy_loss')
        self._mdp = mdp
        self._planner = planner
        self._p = order

    def __call__(self, r_e, r_pi, **kwargs):
        """ Compute the policy loss """
        self._mdp.reward.update_parameters(reward=r_e)
        v_e = self._planner(self._mdp)['V']

        self._mdp.reward.update_parameters(reward=r_pi)
        v_pi = self._planner(self._mdp)['V']

        return np.linalg.norm(v_e - v_pi, ord=self._p)


class RewardLoss(Loss):
    """ Reward loss with respect to a reward function

    L_p = || r_e - r_pi ||_p

    """
    def __init__(self, order=2):
        super(RewardLoss, self).__init__(name='reward_loss')
        self._p = order

    def __call__(self, r_e, r_pi, **kwargs):
        """ Compute the policy loss """
        r_e = np.asarray(r_e)
        r_pi = np.asarray(r_pi)
        assert r_e.shape == r_pi.shape, 'Expecting same shapes'
        return np.linalg.norm(r_e - r_pi, ord=self._p)

