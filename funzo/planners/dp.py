"""
MDP planning using *dynamic programming* methods

    * Policy Iteration (PI)
    * Value  Iteration (VI)

"""

import numpy as np

from six.moves import range

from .base import Planner
from ..utils.validation import check_random_state


__all__ = [
    'PolicyIteration',
    'ValueIteration'
]


class PolicyIteration(Planner):
    """ Policy iteration for computing optimal MDP policy

    Standard Dynamic Programming (DP) using Bellman operator backups

    Parameters
    ------------
    max_iter : int, optional (default: 500)
        Maximum number of iterations of the algorithm
    epsilon : float, optional (default: 1e-08)
        Threshold for policy change in policy evaluation
    verbose : int, optional (default: 4)
        Verbosity level (1-CRITICAL, 2-ERROR, 3-WARNING, 4-INFO, 5-DEBUG)
    random_state : :class:`numpy.RandomState`, optional (default: None)
        Random number generation seed control


    Attributes
    ------------
    _max_iter : int
        Maximum number of iterations of the algorithm
    _epsilon : float
        Threshold for policy change in policy evaluation
    _rng : :class:`numpy.RandomState`
        Random number generator

    """
    def __init__(self, max_iter=200, epsilon=1e-05, verbose=4,
                 random_state=None):
        self._max_iter = max_iter
        self._epsilon = epsilon
        self._rng = check_random_state(random_state)

    def __call__(self, mdp, V_init=None, pi_init=None):
        """ Solve the MDP using policy iteration

        Parameters
        ------------
        mdp : :class:`MDP` variant or derivative
            The MDP to plan on.
        V_init : array-like
            Initial value function
        pi_init : array-like
            Initial policy

        Returns
        --------
        plan : dict
            Dictionary containing the optimal Q, V, pi and cR found

        """
        V = np.zeros(len(mdp.S))
        policy = [self._rng.randint(len(mdp.A)) for _ in range(len(mdp.S))]

        if V_init is not None:
            V = np.array(V_init)

        if pi_init is not None:
            policy = np.array(pi_init)

        cum_R = list()

        stable_policy = False
        R = mdp.R
        T = mdp.T
        while not stable_policy:
            finished = False
            while not finished:
                V_old = np.array(V)
                change = 0.0
                for s in mdp.S:
                    V[s] = R(s, None) + mdp.gamma * \
                        np.sum([p * V[s1] for (p, s1) in T(s, policy[s])])
                change = max(np.fabs(V - V_old))
                if change < self._epsilon:
                    finished = True

            Q = _compute_Q(mdp, V)
            old_policy = np.array(policy)
            policy = np.argmax(Q, axis=0)
            policy_change = max(np.fabs(policy - old_policy))
            if policy_change < 1e-08:
                stable_policy = True

            cum_R.append(np.sum(mdp.R(s, policy[s]) for s in mdp.S))

        result = dict()
        result['pi'] = np.asarray(policy)
        result['V'] = V
        result['Q'] = Q
        result['cR'] = cum_R

        return result


class ValueIteration(Planner):
    """ Value iteration for computing optimal MDP policy

    Standard Dynamic Programming (DP) using Bellman operator backups

    Parameters
    ------------
    max_iter : int, optional (default: 500)
        Maximum number of iterations of the algorithm
    epsilon : float, optional (default: 1e-08)
        Threshold for policy change in policy evaluation
    verbose : int, optional (default: 4)
        Verbosity level (1-CRITICAL, 2-ERROR, 3-WARNING, 4-INFO, 5-DEBUG)

    Attributes
    ------------
    _max_iter : int
        Maximum number of iterations of the algorithm
    _epsilon : float
        Threshold for policy change in policy evaluation

    Returns
    --------
    plan : dict
        Dictionary containing the optimal Q, V and pi found

    """
    def __init__(self, max_iter=200, epsilon=1e-05, verbose=4):
        self._max_iter = max_iter
        self._epsilon = epsilon

    def __call__(self, mdp):
        """ Standard dynamic programming using  value iteration algorithm

        Parameters
        ------------
        mdp : :class:`MDP` variant or derivative
            The MDP to plan on.

        Returns
        --------
        plan : dict
            Dictionary containing the optimal Q, V and pi found

        """
        V = np.zeros(len(mdp.S))
        stable = False
        iteration = 0
        while not stable and iteration < self._max_iter:
            V_old = np.array(V)
            delta = 0
            for s in mdp.S:
                V[s] = mdp.R(s, None) + mdp.gamma * \
                    max([np.sum([p * V_old[s1]
                        for (p, s1) in mdp.T(s, a)])
                        for a in mdp.actions(s)])
                delta = max(delta, np.abs(V[s] - V_old[s]))
            if delta < self._epsilon * (1 - mdp.gamma) / mdp.gamma:
                stable = True

            iteration += 1

        result = dict()
        result['V'] = V
        result['Q'] = _compute_Q(mdp, V)
        result['pi'] = np.argmax(result['Q'], axis=0)
        return result


##############################################################################


def _policy_evaluation(mdp, policy, max_iter=200, epsilon=1e-05):
    """ Compute the value of a policy

    Perform Bellman backups to find the value of a policy for all states in the
    MDP

    """
    finished = False
    iteration = 0
    value = np.zeros(len(mdp.S))
    while iteration < max_iter and not finished:
        v_old = np.array(value)
        delta = 0
        for s in mdp.S:
            # TODO - verify policy[s] or no action
            value[s] = mdp.R(s, None) + mdp.gamma * \
                np.sum([p * value[s1] for (p, s1) in mdp.T(s, policy[s])])
            delta = max(delta, np.fabs(value[s] - v_old[s]))
        if delta < epsilon:
            finished = True

        iteration += 1

    return value


def _expected_utility(mdp, a, s, value):
    """ The expected utility of performing `a` in `s`, using `value` """
    return np.sum([p * mdp.gamma * value[s1] for (p, s1) in mdp.T(s, a)])


def _compute_Q(mdp, value):
    """ Compute the action-value function """
    Q = np.zeros(shape=(len(mdp.A), len(mdp.S)))
    for a in mdp.A:
        for s in mdp.S:
            Q[a][s] = _expected_utility(mdp, a, s, value)
    return Q
