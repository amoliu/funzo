:mod:`funzo.domains.gridworld` 2D Grid world
================================================================

The grid world is a classic Reinforcement learning (RL) domain. The world is
defined by a collection of cells in a 2D grid. Cells can be:
    1. Free
    2. Blocked (obstacle)
    3. Terminal (indicating the goal position) and end of episode

The agent has to move from any start position to the goal position while
avoiding blocked cells. The agent incurs travel cost to encourage shorter paths
and also heavy penalties for getting into blocked cells.

.. automodule:: funzo.domains.gridworld
.. autosummary::

    GridWorld
    GridWorldMDP
    GReward
    GRewardLFA
    GTransition
    GState
    GAction


API
------

.. autoclass:: GridWorld
    :members:
.. autoclass:: GridWorldMDP
    :members:
.. autoclass:: GReward
    :members:
.. autoclass:: GRewardLFA
    :members:
.. autoclass:: GTransition
    :members:
.. autoclass:: GState
    :members:
.. autoclass:: GAction
    :members:
