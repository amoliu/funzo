"""
Base abstract classes for Markov Decision Processes (MDP) and components

These interfaces strive to define a contract for easily implementing relevant
algorithms regardless of the concrete task or domain.
"""

import six
import inspect

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
    'MDPAction',
    'MDPLocalControler',
]


class MDP(six.with_metaclass(ABCMeta, Model)):
    """ Markov Decision Problem (MDP) model

    For general MDPs, states and action can be continuous making it hard to
    efficiently represent them using standard data structures. In the case of
    discrete MDPs, it is straightforward to use array of comparable objects to
    represent the states.

    In the continuous cases, we assume that only a sample of the state and
    action spaces will be used, and these can also be represented a simple
    hash-able data structure (indexed by state or action ids).

    .. note:: This design deliberately leaves out the details of **states** and
        **actions** to be handled by the domain object which includes a
        reference to an MDP object. Additionally, transitions and reward which
        are in general functions are represented as separate *callable*
        objects with references to relevant data needed. This allows a unified
        interface for both **discrete** and **continuous** MDPs and further
        extensions

    Parameters
    ------------
    reward : :class:`RewardFunction` object
        Reward function for the MDP with all the relevant parameters
    transition : :class:`MDPTransition` object
        Represents the transition function for the MDP. All transition relevant
        details such as stochasticity are handled therein.
    discount : float, optional (default: 0.9)
        MDP discount factor in the range [0, 1)
    domain : :class:`Domain` object, optional (default: None)
        The underlying domain (world) on which the MDP operates on

    Attributes
    ------------
    gamma
    _reward : :class:`RewardFunction` object
        Reward function for the MDP with all the relevant parameters
    _transition : :class:`MDPTransition` object
        Represents the transition function for the MDP. All transition relevant
        details such as stochasticity are handled therein.
    _domain : :class:`funzo.domains.Domain` object, optional (default: None)
        The underlying domain (world) on which the MDP operates on. This is
        inferred from the domain context using the ``with`` statement.

    Examples
    ---------
    >>> import numpy as np
    >>> gmap = np.zeros(shape=(3, 3))
    >>> gmap[2, 2] = 2  # set goal with flag 2
    >>> gmap[1, 1] = 1  # set cell blocked
    >>> gmap
    array([[ 0.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  2.]])
    >>> from funzo.domains import GridWorld, GridWorldMDP, GReward, GTransition
    >>> from funzo.planners import PolicyIteration
    >>> planner = PolicyIteration()
    >>> with GridWorld(gmap) as world:
    >>>     R = GReward(9)
    >>>     T = GTransition(wind=0.1)
    >>>     g_mdp = GridWorldMDP(R, T, 0.99)
    >>>     plan = planner.solve(g_mdp)
    >>>     print('Policy: {}'.format(plan['pi']))
    Policy: [0 0 4 3 0 3 0 0 3]

    """

    def __init__(self, reward, transition, discount=0.9, domain=None):
        self._reward = reward
        self._transition = transition
        self.gamma = discount
        self._domain = domain

    def R(self, state, action):
        """ Evaluate the reward function for a (state-action) pair

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
        """ Evaluate the transition function for a (state-action) pair

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
        --------
        a_s : array-like
            The set of available actions at the given state

        """
        raise NotImplementedError('Abstract method')

    def terminal(self, state):
        """ Check if a state is terminal (absorbing) """
        return self._domain.terminal(state)

    @abstractproperty
    def S(self):
        """ Set of states in the MDP in an hashable container """
        raise NotImplementedError('Abstract property')

    @abstractproperty
    def A(self):
        """ Set of actions in the MDP in an hashable container """
        raise NotImplementedError('Abstract property')

    @property
    def reward(self):
        """ Reward function of the MDP """
        return self._reward

    @property
    def gamma(self):
        """ MDP Discount factor """
        return self._discount

    @gamma.setter
    def gamma(self, value):
        """ MDP Discount factor """
        if 0.0 > value >= 1.0:
            raise ValueError('MDP `discount` must be in [0, 1)')
        self._discount = value


########################################################################


class RewardFunction(six.with_metaclass(ABCMeta, Model)):
    """ MDP reward function model

    Rewards are as functions of state and action spaces of MDPs, i.e.

    .. math::

        r: \mathcal{S} \\times \mathcal{A} \longmapsto \mathbb{R}

    Rewards for a state and action pair, :math:`r(s, a)` are accessed via
    the ``__call__`` method while appropriate reward function parameters are
    set in the constructor. In the :class:`MDP` object, this function will be
    called via :class:`MDP.R` function.

    Parameters
    -----------
    domain : :class:`Domain` instance, optional (default: None)
        Object reference to the domain/world of the MDP that the reward is
        to be used
    rmax : float, optional (default: 1.0)
        Upper bound on the reward function

    Attributes
    -----------
    kind
    rmax
    _domain : :class:`Domain` derivative object
        Object reference to the domain of the MDP
    _rmax : float
        Reward upper bound

    """

    def __init__(self, rmax=1.0, domain=None):
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
    def __init__(self, rmax=1.0, domain=None):
        super(TabularRewardFunction, self).__init__(rmax, domain)
        self._R = None

    def update_parameters(self, **kwargs):
        """ Update the internal reward representation parameters """
        if 'reward' in kwargs:
            r = np.asarray(kwargs['reward'])

            if self._R is None:
                self._R = r
            else:
                if r.shape != self._R.shape:
                    raise ValueError('New reward array shape must match'
                                     'reward function dimension')
                self._R = r

    @property
    def kind(self):
        """ Type of reward function (e.g. tabular, LFA) """
        return 'Tabular'

    @abstractmethod
    def __len__(self):
        """ Dimension of the reward function """
        raise NotImplementedError('abstract')


class LinearRewardFunction(six.with_metaclass(ABCMeta, RewardFunction)):
    """ RewardFunction using linear function approximation

    The reward funtion is define as,

    .. math::

        r(s, a) = \sum_i w_i \phi_i(s, a)

    where :math:`\phi_i(s, a)` is a feature defined over state and action
    spaces of the underlying MDP. The ``weights`` are the parameters of the
    model and are usually assumed to sum to 1 to ensure that the reward
    remains bounded, a typical assumption used in most RL planners.

    """

    _template = '_feature_'

    def __init__(self, weights, rmax=1.0, domain=None):
        super(LinearRewardFunction, self).__init__(rmax, domain)
        self._weights = np.asarray(weights)
        assert self._weights.ndim == 1, 'Weights must be 1D arrays'

    def update_parameters(self, **kwargs):
        """ Update the weights parameters of the reward function model """
        if 'reward' in kwargs:
            w = np.asarray(kwargs['reward'])
            assert w.shape == self._weights.shape,\
                'New weight array size must match reward function dimension'
            self._weights = w

    @property
    def kind(self):
        """ Type of reward function (e.g. tabular, LFA) """
        return 'LFA'

    @abstractmethod
    def phi(self, state, action):
        """ Evaluate the reward features for state-action pair """
        raise NotImplementedError('abstract')

    def __len__(self):
        """ Dimension of the reward function in the case of LFA """
        # - count all class members named '_feature_{x}'
        dim = 0
        for name in self.__class__.__dict__:
            item = getattr(self.__class__, name)
            if inspect.ismethod(item):
                if item.__name__.startswith(self._template):
                    dim += 1
        return dim

########################################################################


class MDPTransition(six.with_metaclass(ABCMeta, Model)):
    """ A MDP single step transition function

    .. math::

        T: \mathcal{S} \\times \mathcal{A} \\times \mathcal{S}
        \longmapsto \mathbb{R}

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
        Object reference to the domain of the underlying MDP

    """

    def __init__(self, domain=None):
        self._domain = domain

    @abstractmethod
    def __call__(self, state, action, **kwargs):
        """ Execute the transition function

        Run the controller at `state` using `action` with optional parameters
        given in `kwargs`

        """
        raise NotImplementedError('Abstract method')


class MDPLocalControler(six.with_metaclass(ABCMeta, Model)):
    """ A MDP local controller

    Representing multiple step transition, which can be interpreted as a
    Markov option with only one possible terminal state.

    """
    def __init__(self, domain):
        self._domain = domain

    @abstractmethod
    def __call__(self, state, action, duration, **kwargs):
        """ Execute the local controller """
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def trajectory(self, source, target, **kwargs):
        """ Compute the trajectory/policy between two states """
        raise NotImplementedError('Abstract')

########################################################################


class MDPState(six.with_metaclass(ABCMeta, Model, Hashable)):
    """ MDP State

    A state in an MDP with all the relevant domain specific data. Such data
    is used in the reward function and transition function for computing
    various quantities. Every state must be a comparable object with an id

    """

    def __init__(self, state_id):
        self._id = state_id

    @abstractmethod
    def __hash__(self):
        """ State hash function """
        raise ValueError('Implement a hash function')

    @property
    def id(self):
        """ State unique identifier """
        return self._id

    @abstractmethod
    def __eq__(self, other):
        """ State comparator function """
        raise NotImplementedError('Implement equality of states')


class MDPAction(six.with_metaclass(ABCMeta, Model, Hashable)):
    """ MDP Action

    An action in an MDP with all the relevant domain specific data. Such data
    is used in the reward function and transition function for computing
    various quantities.

    """

    def __init__(self, action_id):
        self._id = action_id

    @abstractmethod
    def __hash__(self):
        """ Action hash function """
        raise ValueError('Implement a hash function')

    @property
    def id(self):
        """ Action unique identifier """
        return self._id

    @abstractmethod
    def __eq__(self, other):
        """ Action comparator function """
        raise NotImplementedError('Implement equality of actions')
