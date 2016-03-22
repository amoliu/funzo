"""
Base interfaces for Markov Decision Processes (MDP)

These interfaces strive to define a contract for easily implementing relevant
algorithms regardless of the concrete task or domain.


"""


import six

from abc import ABCMeta
from abc import abstractmethod, abstractproperty
from collections import Hashable

import numpy as np

from ..base import Model


__all__ = [
    'MDP',
    'RewardFunction',
    'LinearRewardFunction',
    'TabularRewardFunction',
    'MDPTransition',
    'MDPState',
    'MDPAction'
]


class MDP(six.with_metaclass(ABCMeta, Model)):
    """ Markov Decision Process (MDP) model

    For general MDPs, states and action can be continuous making it hard to
    efficiently represent them using standard data structures. In the case of
    discrete MDPs, it is straightforward to use array of comparable objects to
    represent the states.

    In the continuous cases, we assume that only a sample of the state and
    action spaces will be used, and these can also be represented a simple
    hashable data structure (indexed by state or action ids).

    Parameters
    ------------
    domain : :class:`Domain` object
        The underlying domain (world) on which the MDP operates on
    discount : float
        MDP discount factor in the range [0, 1)
    reward : :class:`RewardFunction` object
        Reward function for the MDP with all the relevant parameters
    transition : :class:`MDPTransition` object
        Represents the transition function for the MDP. All transition relevant
        details such as stochasticity are handled therein.

    Attributes
    ------------
    _domain : :class:`Domain` object
        The underlying domain (world) on which the MDP operates on
    gamma : float
        MDP discount factor
    _reward : :class:`RewardFunction` object
        Reward function for the MDP with all the relevant parameters
    _transition : :class:`MDPTransition` object
        Represents the transition function for the MDP. All transition relevant
        details such as stochasticity are handled therein.


    Notes
    ------
    This design deliberately leaves out the details of *states* and *actions*
    to be handled by the domain object which includes a reference to an MDP
    object. Additionally, transitions and reward which are in general functions
    are represented as separate *callable* objects with references to relevant
    data needed. This allows a unified interface for both *discrete* and
    *continuous* MDPs and further extensions

    """

    def __init__(self, domain, reward, transition, discount=0.9):
        self._domain = domain
        self._reward = reward
        self._transition = transition
        self.gamma = discount

    def R(self, state, action):
        """ Evaluate the reward function

        The reward for performing `action` in `state`. Additional reward
        parameters can be included in the definition of the reward class

        Parameters
        -----------
        state : int
            state id for a state object in the MDP
        action : int
            id for an MDP action

        Returns
        --------
        reward : float
            A real valued reward signal

        """
        return self._reward(state, action)

    def T(self, state, action):
        """ Evaluate the transition function

        Perform a transition from a state using the action specified. The
        result is all reachable states with their respective "reach"
        probabilities. In the case of deterministic dynamics, the result will
        contain only one of the reachable states.

        Parameters
        -----------
        state : int
            state id for a state object in the MDP
        action : int
            id for an MDP action

        Returns
        --------
        next_states : array-like
            Array of all reachable states and their transition probabilities
            i.e. :math:`\{(p, s') \\forall s' \in T(s, a, \cdot) \}`

        """
        return self._transition(state, action)

    @abstractmethod
    def actions(self, state):
        """ Get actions available at a state

        Set the set of actions available at a state. The dynamic model, T then
        together with the policy induce a probability distribution over this
        set.

        Parameters
        -----------
        state : int
            state id for a state object in the MDP

        Returns
        -------
        a_s : array-like, shape (|A|_s,)
            The set of available actions at the given state

        """
        raise NotImplementedError('Abstract method')

    def terminal(self, state):
        """ Check if a state is terminal (absorbing state) """
        return self._domain.terminal(state)

    @abstractproperty
    def S(self):
        """ States of the MDP in an hashable container """
        raise NotImplementedError('Abstract property')

    @abstractproperty
    def A(self):
        """ Actions of the MDP in an hashable container """
        raise NotImplementedError('Abstract property')

    @property
    def reward(self):
        """ Accessor for the underlying reward object """
        return self._reward

    @property
    def gamma(self):
        return self._discount

    @gamma.setter
    def gamma(self, value):
        if 0.0 > value >= 1.0:
            raise ValueError('MDP `discount` must be in [0, 1)')
        self._discount = value


########################################################################


class RewardFunction(six.with_metaclass(ABCMeta, Model)):
    """ Markov decision process reward  function interface

    Rewards are as functions of state and action spaces of MDPs, i.e.

    .. math::

        r: \mathcal{S} \\times \mathcal{A} \longrightarrow \mathbb{R}

    Rewards are accessed via the `__call__` method with appropriate
    parameters.


    Parameters
    -----------
    domain : :class:`Domain` derivative object
        Object reference to the domain of the MDP that the reward is
        to be used
    rmax : float, optional (default: 1.0)
        Upper bound on the reward function

    Attributes
    -----------
    _domain : :class:`Domain` derivative object
        Object reference to the domain of the MDP
    _rmax : float
        Reward upper bound

    """

    def __init__(self, domain, rmax=1.0):
        # keep a reference to parent MDP to get access to domain and dynamics
        self._domain = domain
        self._rmax = rmax

    @abstractmethod
    def __call__(self, state, action):
        """ Evaluate the reward function for the (state, action) pair """
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def __len__(self):
        """ Dimension of the reward function """
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def update_parameters(self, **kwargs):
        """ Update the parameters of the reward function model """
        raise NotImplementedError('Abstract method')

    @abstractproperty
    def kind(self):
        """ Type of reward function (e.g. tabular, LFA) """
        raise NotImplementedError('Abstract property')

    @property
    def rmax(self):
        """ Reward upper bound """
        return self._rmax


class TabularRewardFunction(six.with_metaclass(ABCMeta, RewardFunction)):
    """ Reward function with a tabular representation

    A basic reward function with full tabular representation, mainly suitable
    for discrete and small sized domains. i.e. :math:`r(s, a) = R[s, a]` where
    :math:`R` is a tensor.

    """
    def __init__(self, domain, n_s, n_a=None, rmax=1.0):
        # If n_a is 0, assume only state based reward function
        super(TabularRewardFunction, self).__init__(domain, rmax)
        assert n_s > 0, 'Number of states must be greater than 0'
        self._n_s = n_s

        if n_a is not None:
            assert n_a > 0, 'Number of actions must be greater than 0'
            self._n_a = n_a
            self._R = np.zeros(self._n_s, self._n_a)
        else:
            self._R = np.zeros(self._n_s)

    def update_parameters(self, **kwargs):
        """ Update the internal reward representation parameters """
        if 'reward' in kwargs:
            r = np.asarray(kwargs['reward'])
            assert r.shape == self._R.shape,\
                'New reward array shape must match reward function dimension'
            self._R = r

    @property
    def kind(self):
        """ Type of reward function (e.g. tabular, LFA) """
        return 'Tabular'

    def __len__(self):
        """ Dimension of the reward function """
        return self._R.size


class LinearRewardFunction(six.with_metaclass(ABCMeta, RewardFunction)):
    """ RewardFunction using Linear Function Approximation

    The reward is given by;

    .. math:

            r(s, a) = \sum_i w_i f_i(s, a)

    where :math:`f_(s, a)` is a reward feature defined over state and action
    spaces of the underlying MDP. The *weights* are the parameters of the
    model as usually sum to 1 to ensure that the reward remains bounded, a
    typical assumption in RL.

    """

    _template = '_feature_'

    def __init__(self, domain, weights, rmax=1.0):
        super(LinearRewardFunction, self).__init__(domain, rmax)
        self._weights = np.asarray(weights)
        assert self._weights.ndim == 1, 'Weights must be 1D arrays'

        # ensure reward is bounded by normalizing the weights
        self._weights /= (self._weights.max() - self._weights.min())

    def update_parameters(self, **kwargs):
        """ Update the weights parameters of the reward function model """
        if 'reward' in kwargs:
            w = np.asarray(kwargs['reward'])
            assert w.shape == self._weights.shape,\
                'New weight array size must match reward function dimension'
            w /= (w.max() - w.min())
            self._weights = w

    @property
    def kind(self):
        """ Type of reward function (e.g. tabular, LFA) """
        return 'LFA'

    def __len__(self):
        """ Dimension of the reward function in the case of LFA """
        # - count all class members named '_feature_{x}'
        features = self.__class__.__dict__
        dim = sum([f[0].startswith(self._template) for f in features])
        return dim

########################################################################


class MDPTransition(six.with_metaclass(ABCMeta, Model)):
    """ A MDP transition function

    A generic way of representing MDP transition operation for both discrete
    and continuous spaces. A T function simply takes and `action` at a given
    `state` and executes it based on the transition properties (which could
    include stochasticity, etc)

    Parameters
    -----------
    domain : :class:`Domain` derivative object
        Object reference to the domain of the MDP that the controller is
        to be used on

    Attributes
    -----------
    _domain : :class:`Domain` derivative object
        Object reference to the domain of the MDP that the controller is
        to be used on

    """

    def __init__(self, domain):
        self._domain = domain

    @abstractmethod
    def __call__(self, state, action, **kwargs):
        """ Execute the transition function

        Run the controller at `state` using `action` with optional parameters
        given in `kwargs`

        """
        raise NotImplementedError('Abstract method')


########################################################################


class MDPState(six.with_metaclass(ABCMeta, Model, Hashable)):
    """ State on an MDP

    A state in an MDP with all the relevant domain specific data. Such data
    is used in the reward function and transition function for computing
    various quantities. Every state must be a comparable object with an id

    """

    def __init__(self, state_id):
        self._id = state_id

    @abstractmethod
    def __hash__(self):
        raise ValueError('Implement a hash function')

    @property
    def id(self):
        return self._id

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError('Implement equality of states')


class MDPAction(six.with_metaclass(ABCMeta, Model, Hashable)):
    """ Action in an MDP

    An action in an MDP with all the relevant domain specific data. Such data
    is used in the reward function and transition function for computing
    various quantities.

    """

    def __init__(self, action_id):
        self._id = action_id

    @abstractmethod
    def __hash__(self):
        raise ValueError('Implement a hash function')

    @property
    def id(self):
        return self._id

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError('Implement equality of actions')
