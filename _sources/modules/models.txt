:mod:`funzo.models`
====================================
.. automodule:: funzo.models

The core models for reinforcement learning. The implementation here strives to
be as general as possible to allow for easy switching of algorithms for working
on these models.

Markov decision processes (MDP)
-----------------------------------
Strictly speaking, Markov decision *processes* only contain state space,
action space, transition function and possible discounting factor. By adding a
reward function they become Markov decision *problems*. We however use MDP to
refer to both interchangeably.

.. autosummary::

   MDP
   MDPState
   MDPAction
   MDPTransition
   MDPLocalController
   RewardFunction
   TabularRewardFunction
   LinearRewardFunction


API
------
.. autoclass:: MDP
    :members:
.. autoclass:: MDPState
    :members:
.. autoclass:: MDPAction
    :members:
.. autoclass:: MDPTransition
    :members:
.. autoclass:: MDPLocalController
    :members:
.. autoclass:: RewardFunction
    :members:
.. autoclass:: TabularRewardFunction
    :members:
.. autoclass:: LinearRewardFunction
    :members:
