""" Base and mixin classes for planners for MDPs """


import six

from abc import ABCMeta
from abc import abstractmethod

from ..base import Model


class Planner(six.with_metaclass(ABCMeta, Model)):
    """ A planner for MDPs

    A planner computes a policy and optionally value and quality functions
    given the specification of the MDP model.

    """

    @abstractmethod
    def solve(self, mdp, V_init=None, pi_init=None):
        """ Run the planner on a MDP to get the policy """
        raise NotImplementedError('Abstract method')
