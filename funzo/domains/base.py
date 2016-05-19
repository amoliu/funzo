"""
Base interfaces for domains (worlds) which contain all relavant information
about the states and actions in MDPs used by the different functions (reward
controller, transition)

"""

import six

from abc import ABCMeta
from abc import abstractmethod

from ..base import Model


__all__ = ['Domain', 'model_domain']


class Domain(six.with_metaclass(ABCMeta, Model)):

    """ Domain (World) interface

    Domains expand on the abstraction in the MDP, they represent the concrete
    details of the task. i.e. the environment.

    """

    @abstractmethod
    def terminal(self, state):
        """ Check if a state is terminal (absorbing state) """
        raise NotImplementedError('Abstract method')

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
        raise NotImplementedError('This method is abstract')

    @abstractmethod
    def in_domain(self, state):
        """ Check if a state is within domain's bounds """
        raise NotImplementedError('Abstract')

    def __enter__(self):
        type(self).get_domains().append(self)
        return self

    def __exit__(self, typ, value, traceback):
        type(self).get_domains().pop()

    @classmethod
    def get_domains(cls):
        if not hasattr(cls, "domains"):
            cls.domains = []

        return cls.domains

    @classmethod
    def get_domain(cls):
        """Return the deepest domain on the stack."""
        try:
            return cls.get_domains()[-1]
        except IndexError:
            raise TypeError("No domain on domain stack")


def model_domain(model=None, domain_type=Domain):
    """ Get the domain of the model """
    if model is None:
        return domain_type.get_domain()
    return model
