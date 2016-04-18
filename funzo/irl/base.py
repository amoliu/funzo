"""
Inverse reinforcement learning algorithms base modules

"""
from abc import ABCMeta, abstractmethod

import six
import numpy as np

from ..base import Model


class IRLSolver(six.with_metaclass(ABCMeta, Model)):
    """ IRL algorithm interface """
    def __init__(self, mdp_planner=None):
        self._mdp_planner = mdp_planner

    @abstractmethod
    def solve(self, demos, mdp=None):
        """ Solve the IRL problem using the MDP and demonstrations """
        raise NotImplementedError('Abstract interface')

    def _solve_mdp(self, mdp, r, V_init=None, pi_init=None):
        """ Solve and MDP using a given reward function """
        mdp.reward.update_parameters(reward=r)
        plan = self._mdp_planner(mdp, V_init, pi_init)
        return plan


########################################################################
# Loss functions
########################################################################

class Loss(six.with_metaclass(ABCMeta, Model)):
    """ A loss function for evaluating progress of IRL algorithms """

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, r_e, r_pi, **kwargs):
        # TODO - May enforce implementation with ufuncs ?
        raise NotImplementedError('abstract')


class PolicyLoss(Loss):
    """ Policy loss with respect to a reward function

    L_p = || V^*(r) - V^{\pi}(r) ||_p

    Roughly corresponds to apprenticeship learning

    """
    def __init__(self, mdp, planner, order=2):
        super(PolicyLoss, self).__init__(name='policy_loss')
        self._mdp = mdp
        self._planner = planner
        self._p = order

        # caching
        self._ve = None
        self._pi_e = None
        self._vpi = None
        self._pi_pi = None

    def __call__(self, r_e, r_pi, **kwargs):
        """ Compute the policy loss """
        self._mdp.reward.update_parameters(reward=r_e)
        plan_e = self._planner(self._mdp, self._ve, self._pi_e)
        self._ve = plan_e['V']
        self._pi_e = plan_e['pi']

        self._mdp.reward.update_parameters(reward=r_pi)
        plan_pi = self._planner(self._mdp, self._vpi, self._pi_pi)
        self._vpi = plan_pi['V']
        self._pi_pi = plan_pi['pi']

        return np.linalg.norm(self._ve - self._vpi, ord=self._p)


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
        if r_e.shape != r_pi.shape:
            raise ValueError('Expert and learned reward dimensions mismatch')
        return np.linalg.norm(r_e - r_pi, ord=self._p)
