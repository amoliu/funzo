"""
Base interfaces for Markov Decision Processes (MDP)

These interfaces strive to define a contract for easily implementing relavant
algorithms regardless of the concrete task or domain.


"""


import six

from abc import ABCMeta
from abc import abstractmethod, abstractproperty


from ..base import Model


class MDP(Model):
    """ Markov Decision Process Model

    Parameters
    ------------
    discount : float
        MDP discount factor
    reward : :class:`MDPReward` object
        Reward function for the MDP with all the relavant parameters
    transition : :class: `MDPController` object
        Represents the transition function for the MDP. All transition relevant
        details such as stochaticity are handled therein.

    """

    def __init__(self, discount, reward, transition):
        self.gamma = discount
        self._reward = reward  # keep a reference to reward function object
        self._transition = transition

    @abstractproperty
    def S(self):
        raise NotImplementedError('Abstract property')

    @abstractproperty
    def A(self):
        raise NotImplementedError('Abstract property')

    @abstractmethod
    def R(self, state, action):
        """ Reward function

        The reward for performing `action` in `state`. Additional reward
        parameters can be included in the definition of the reward class

        Parameters
        -----------
        state : State
            A state in the MDP
        action : Action
            MDP action

        Returns
        --------
        reward : float
            A real valued reward signal

        """
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def T(self, state, action):
        """ Transition from `state` with `action`

        Perform a transion from a state using the action specified. The result
        is all reachable states with their respective "reach" probabilities. In
        the case of deteministic dynamics, the result will contain only one of
        the reachable states.

        Parameters
        -----------
        state : State
            A state in the MDP
        action : Action
            MDP action

        Returns
        --------
        next_states : array
            Array of all reachable states and their transition probabilities
            i.e. :math:`\{(p, s') \\forall s' \in T(s, a, \cdot) \}`

        """
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def terminal(self, state):
        """ Check if a state is terminal (absorbing state) """
        raise NotImplementedError('Abstract method')

    @property
    def gamma(self):
        return self._discount

    @gamma.setter
    def gamma(self, value):
        assert 0.0 <= value < 1.0, 'MDP `discount` must be in [0, 1)'
        self._discount = value


########################################################################


class MDPReward(six.with_metaclass(ABCMeta, Model)):
    """ Markov decision process reward  function interface

    Rewards are as functions of state and action spaces of MDPs, i.e.

    .. math::
        :label: reward

        r: \mathcal{S} \\times \mathcal{A} \longrightarrow \mathbb{R}

    Rewards are accessed via the `__call__` method with apppropriate
    parameters.


    Parameters
    -----------
    world : `class` object
        Object reference to the domain of the MDP that the reward is
        to be used

    Attributes
    -----------
    _world : `class` object
        Object reference to the domain of the MDP that the reward is
        to be used

    Note
    -------
    The dimension of the reward in case of linear function representation is
    computed based on a convention of reward function names defined by the
    `_template` tag

    """

    _template = '_feature_'

    def __init__(self, world):
        # keep a reference to parent MDP to get access to domain and dynamics
        self._world = world

    @abstractmethod
    def __call__(self, state, action):
        """ Evaluate the reward function for the (state, action) pair

        Compute :math:`r(s, a) = \sum_i w_i f_i(s, a)`
        if  :math:`f` is a (linear )function approximation for the reward
        parameterized by :math:`w`

        Otherwise, return the tabular reward for the given state.

        """
        raise NotImplementedError('Abstract method')

    @property
    def dim(self):
        """ Dimension of the reward function in the case of LFA """
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


########################################################################


class MDPController(six.with_metaclass(ABCMeta, Model)):
    """ A MDP controller

    A generic way of representing MDP transition operation for both discrete
    and continuous spaces. A controller simply takes and `action` at a given
    `state` and executes it based on the controller properties (which could
    include stochaticity, etc)

    Parameters
    -----------
    world : `class` object
        Object reference to the domain of the MDP that the controller is
        to be used on
    kind : str
        Controller type (descriptive tag)

    Attributes
    -----------
    _world : `class` object
        Object reference to the domain of the MDP that the controller is
        to be used on
    kind : str
        Controller type (descriptive tag)


    """

    def __init__(self, world, kind='abstract'):
        self._world = world
        self.kind = kind

    @abstractmethod
    def __call__(self, state, action, **kwargs):
        """ Execute a controller

        Run the controller at `state` using `action` with optional parameters
        given in `kwargs`

        """
        raise NotImplementedError('Abstract method')
