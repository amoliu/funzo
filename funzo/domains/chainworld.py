"""
ChainWorld Domain

Agent exists in a finite chain and can move left or right. The goal is on
the right most end of the chain, which marks the end of an episode.


"""

from __future__ import division

import numpy as np

from six.moves import range
from matplotlib.patches import Circle

from .base import Domain, model_domain
from ..models.mdp import MDP
from ..models.mdp import TabularRewardFunction
from ..models.mdp import MDPTransition, MDPState, MDPAction


__all__ = [
    'ChainMDP',
    'ChainWorld',
    'ChainAction',
    'ChainState',
    'ChainTransition',
    'ChainReward',
]


class ChainState(MDPState):
    """ A state in the ChainWorld MDP """
    def __init__(self, state_id):
        super(ChainState, self).__init__(state_id)

    def __hash__(self):
        """ ChainWorld state hash function """
        return (self.id).__hash__()

    def __eq__(self, other):
        """ ChainWorld state comparator function """
        return self.id == other.id


class ChainAction(MDPAction):
    """ An action in the ChainWorld MDP """
    def __init__(self, action_id, direction):
        super(ChainAction, self).__init__(action_id)
        if direction not in [-1, 1]:
            raise ValueError('ChainWorld action can only be -1 or 1')
        self.direction = direction

    def __hash__(self):
        """ ChainWorld action hash function """
        return (self.id).__hash__()

    def __eq__(self, other):
        """ ChainWorld action comparator function """
        return self.id == other.id


class ChainTransition(MDPTransition):
    """ Transition function for ChainWorld """
    def __init__(self, domain=None):
        super(ChainTransition, self).__init__(domain)
        self._domain = model_domain(domain, ChainWorld)

    def __call__(self, state, action, **kwargs):
        """ Execute the transition function """
        state_ = self._domain.states[state]
        action_ = self._domain.actions[action]
        next_state = state_.id + action_.direction
        if self._domain.in_domain(next_state):
            return [(1.0, next_state)]
        return [(1.0, state)]


class ChainReward(TabularRewardFunction):
    """ Reward function for ChainWorld """
    def __init__(self, domain=None):
        super(ChainReward, self).__init__(domain)
        self._domain = model_domain(domain, ChainWorld)

        R = np.zeros(len(self))
        R[-1] = 1
        self.update_parameters(reward=R)

        self._T = ChainTransition(domain=domain)

    def __call__(self, state, action):
        """ Evaluate reward function """
        if action is None:
            return self._R[state]
        s_p = self._T(state, action)[0][1]
        if s_p == state:
            return -10.0
        return self._R[s_p]

    def __len__(self):
        return len(self._domain.states)


#############################################################################


class ChainWorld(Domain):
    """ ChainWorld """
    def __init__(self, num_states=5):
        self.states = dict()
        for i in range(num_states):
            self.states[i] = ChainState(i)

        self.actions = {
            0: ChainAction(0, 1),
            1: ChainAction(1, -1),
        }

    def terminal(self, state):
        """ Check if a state is terminal (absorbing state) """
        return state == (len(self.states) - 1)

    def visualize(self, ax, **kwargs):
        """ visualize the ChainWorld domain

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
        return self._plot_world(ax)

    def in_domain(self, state):
        return state >= 0 and state < len(self.states)

    def _plot_world(self, ax):
        for s in self.states:
            loc = (s + 1, 2)
            face = 'gray' if self.terminal(s) else 'w'
            ax.add_artist(Circle(loc, radius=0.3, color=face,
                          ec='r', lw=1.5, aa=True))
            text = 's' + str(s + 1)
            ax.text(loc[0], loc[1], text, ha="center", size=14)

        ax.set_xlim([0, len(self.states) + 1])
        ax.set_ylim([1, 3])
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def show_policy(self, ax, policy=None):
        """
        Show a policy on plot
        """
        if policy is not None:
            assert len(policy) == len(self.states),\
                'Policy not compatible with state space dimensions'
            for s in range(policy.shape[0]):
                a = policy[s]
                text = 'G'
                if self.actions[int(a)].direction == 1:
                    text = '$\\Rightarrow$'
                elif self.actions[int(a)].direction == -1:
                    text = '$\\Leftarrow$'

                ss = (s + 1, 2)
                ax.text(ss[0] + (1 / 2.), ss[1],
                        text, ha="center", size=14)


class ChainMDP(MDP):
    """ ChainWorld MDP """
    def __init__(self, reward, transition, discount=0.9, domain=None):
        super(ChainMDP, self).__init__(reward,
                                       transition,
                                       discount,
                                       domain)
        self._domain = model_domain(domain, ChainWorld)

    @property
    def S(self):
        """ States of the ChainWorld MDP in an indexable container """
        return self._domain.states.keys()

    @property
    def A(self):
        """ Actions of the ChainWorld MDP in an indexable container """
        return self._domain.actions.keys()

    def actions(self, state):
        """ Set of actions that can be performed in this state."""
        return self.A.keys()
