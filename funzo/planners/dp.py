"""
MDP planning using *dynamic programming* methods

    * Policy Iteration (PI)
    * Value  Iteration (VI)

"""

import logging
import copy

from tqdm import tqdm
from six.moves import range
import numpy as np

from ..utils.validation import check_random_state

logger = logging.getLogger(__name__)

# TODO - define a planner interface


def policy_iteration(mdp, max_iter=200, epsilon=1e-05, verbose=4,
                     random_state=None):
    """ Policy iteraction for computing optimal MDP policy

    Standard Dynamic Programming (DP) using Bellman operator backups

    Parameters
    ------------
    mdp : :class:`MDP` variant or derivative
        The MDP to plan on.
    max_iter : int, optional (default: 500)
        Maximum number of iterations of the algorithm
    epsilon : float, optional (default: 1e-08)
        Threshold for policy change in policy evaluation
    verbose : int, optional (default: 4)
        Verbosity level (1-CRITICAL, 2-ERROR, 3-WARNING, 4-INFO, 5-DEBUG)
    random_state : :class:`numpy.RandomState`, optional (default: None)
        Random number generation seed control

    Returns
    --------
    plan : dict
        Dictionary containing the optimal Q, V and pi found

    """
    logging.basicConfig(level=verbose)

    V = np.zeros(len(mdp.S))
    rng = check_random_state(random_state)
    policy = [rng.randint(len(mdp.A)) for _ in range(len(mdp.S))]
    iteration = 0
    for iteration in tqdm(range(0, max_iter)):
        V = _policy_evaluation(mdp, policy, max_iter, epsilon)

        # policy improvement
        unchanged = True
        for s in mdp.S:
            a = np.argmax([_expected_utility(mdp, a, s, V)
                          for a in mdp.actions(s)])
            if a != policy[s]:
                policy[s] = a
                unchanged = False
        if unchanged:
            break

        # logger.debug('PI, iteration: %s' % iteration)

    result = dict()
    result['pi'] = np.asarray(policy)
    result['V'] = V
    result['Q'] = _compute_Q(mdp, V)
    return result


def value_iteration(mdp, max_iter=200, epsilon=1e-05, verbose=4):
    """ Value iteraction for computing optimal MDP policy

    Standard Dynamic Programming (DP) using Bellman operator backups

    Parameters
    ------------
    mdp : :class:`MDP` variant or derivative
        The MDP to plan on.
    max_iter : int, optional (default: 500)
        Maximum number of iterations of the algorithm
    epsilon : float, optional (default: 1e-08)
        Threshold for policy change in policy evaluation
    verbose : int, optional (default: 4)
        Verbosity level (1-CRITICAL, 2-ERROR, 3-WARNING, 4-INFO, 5-DEBUG)


    Returns
    --------
    plan : dict
        Dictionary containing the optimal Q, V and pi found

    """
    logging.basicConfig(level=verbose)

    V = np.zeros(len(mdp.S))
    stable = False
    iteration = 0
    while not stable and iteration < max_iter:
        V_old = copy.deepcopy(V)
        delta = 0
        for s in mdp.S:
            V[s] = mdp.R(s, None) + mdp.gamma * \
                max([np.sum([p * V_old[s1]
                    for (p, s1) in mdp.T(s, a)])
                    for a in mdp.actions(s)])
            delta = max(delta, np.abs(V[s] - V_old[s]))
        if delta < epsilon * (1 - mdp.gamma) / mdp.gamma:
            stable = True

        iteration += 1
        logger.info('VI, iter: %s' % iteration)

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
        v_old = copy.deepcopy(value)
        delta = 0
        for s in mdp.S:
            value[s] = mdp.R(s, None) + mdp.gamma * \
                sum([p * value[s1] for (p, s1) in mdp.T(s, policy[s])])
            delta = max(delta, np.abs(value[s] - v_old[s]))
        if delta < epsilon:
            finished = True

        iteration += 1
        # logger.info('Policy evaluation, iter: %s' % iteration)

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
