"""
Exact MDP planning using *exact* methods

    * Policy Iteration (PI)
    * Value  Iteration (VI)
    * Linear Programming (LP)

Assumptions
=============
Both the state and action space of the MDP can be compactly represented, e.g.
in discrete case with small to medium size MDPs

"""


import numpy as np


def policy_iteration(mdp):
    pass


def value_iteration(mdp, epsilon=1e-05):
    """ Value iteraction for computing optimal MDP policy

    Standard Dynamic Programming (DP) using Bellman operator backups

    Parameters
    ------------
    mdp : :class: of `MDP` variant
        The MDP to plan on.

    Returns
    --------
    plan : dict
        Dictionary containing the optimal Q, V and pi found

    """
    # 1. check that the mdp is enumerable in S and A
    # 2. other sanity checks

    V = np.zeros(len(mdp.S))
    policy = [np.random.randint(len(mdp.A)) for _ in range(len(mdp.S))]
    Q = np.zeros(shape=(len(mdp.S), len(mdp.A)))

    return {'V': V, 'pi': policy, 'Q': Q}


def _policy_evaluation(mdp, policy):
    pass


def _policy_improvement(mdp, Q):
    pass
