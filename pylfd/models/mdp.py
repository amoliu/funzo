"""
Base interfaces for Markov Decision Processes (MDP)

These interfaces strive to define a contract for easily implementing relavant
algorithms regardless of the concrete task or domain.


"""


import six

from abc import ABCMeta
from abc import abstractmethod, abstractproperty


from ..base import Model


__all__ = [
    'MDP',
    'MDPReward',
    'MDPRewardLFA',
    'MDPTransition'
]


class MDP(Model):
    """ Markov Decision Process Model


    For general MDPs, states and action can be continuous making it hard to
    efficiently represent them using standard data strcutures. In the case of
    discrete MDPs, it is straightforward to develop indexable data strcutures
    to contain all possible states and actions (evne though these may be huge).
    In the continuous cases, we assume that only a sample of the state and
    action spaces will be used, and these can also be represented with relavant
    indexable data strcutures.

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

    def R(self, state, action):
        """ Reward function

        The reward for performing `action` in `state`. Additional reward
        parameters can be included in the definition of the reward class

        Parameters
        -----------
        state : int
            A state id in the MDP, used to index into the relevant state. The
            representation of state is irrelevant as long as `self.S[state]`
            returns a meaningful state on which the reward can be computed
        action : int
            MDP action id analogous the state id described above.

        Returns
        --------
        reward : float
            A real valued reward signal

        """
        return self._reward(state, action)

    def T(self, state, action):
        """ Transition from `state` with `action`

        Perform a transition from a state using the action specified. The
        result is all reachable states with their respective "reach"
        probabilities. In the case of deteministic dynamics, the result will
        contain only one of the reachable states.

        Parameters
        -----------
        state : int
            A state id in the MDP, used to index into the relevant state. The
            representation of state is irrelevant as long as `self.S[state]`
            returns a meaningful state on which the transition can be computed
        action : int
            MDP action id analogous the state id described above.

        Returns
        --------
        next_states : array
            Array of all reachable states and their transition probabilities
            i.e. :math:`\{(p, s') \\forall s' \in T(s, a, \cdot) \}`

        """
        return self._transition(state, action)

    @abstractmethod
    def terminal(self, state):
        """ Check if a state is terminal (absorbing state) """
        raise NotImplementedError('Abstract method')

    @abstractproperty
    def S(self):
        """ States of the MDP in an indexable container """
        raise NotImplementedError('Abstract property')

    @abstractproperty
    def A(self):
        """ Actions of the MDP in an indexable container """
        raise NotImplementedError('Abstract property')

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

        r: \mathcal{S} \\times \mathcal{A} \longrightarrow \mathbb{R}

    Rewards are accessed via the `__call__` method with apppropriate
    parameters.


    Parameters
    -----------
    domain : `class` object
        Object reference to the domain of the MDP that the reward is
        to be used

    Attributes
    -----------
    _domain : `class` object
        Object reference to the domain of the MDP that the reward is
        to be used

    Note
    -------
    The dimension of the reward in case of linear function representation is
    computed based on a convention of reward function names defined by the
    `_template` tag

    """

    def __init__(self, domain):
        # keep a reference to parent MDP to get access to domain and dynamics
        self._domain = domain

    @abstractmethod
    def __call__(self, state, action):
        """ Evaluate the reward function for the (state, action) pair """
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def __len__(self):
        """ Dimension of the reward function """
        raise NotImplementedError('Abstract method')


class MDPRewardLFA(six.with_metaclass(ABCMeta, MDPReward)):
    """ MDPReward using Linear Function Approximation

    The reward is given by;

    .. math:

            r(s, a) = \sum_i w_i f_i(s, a)

    where :math:`f_(s, a)` is a reward feature defined over state and action
    spaces of the underlying MDP

    """

    _template = '_feature_'

    def __init__(self, domain):
        super(MDPRewardLFA, self).__init__(domain)

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
    and continuous spaces. A controller simply takes and `action` at a given
    `state` and executes it based on the controller properties (which could
    include stochaticity, etc)

    Parameters
    -----------
    domain : `class` object
        Object reference to the domain of the MDP that the controller is
        to be used on

    Attributes
    -----------
    _domain : `class` object
        Object reference to the domain of the MDP that the controller is
        to be used on

    """

    def __init__(self, domain):
        self._domain = domain

    @abstractmethod
    def __call__(self, state, action, **kwargs):
        """ Execute a controller

        Run the controller at `state` using `action` with optional parameters
        given in `kwargs`

        """
        raise NotImplementedError('Abstract method')
