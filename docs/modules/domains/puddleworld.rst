:mod:`funzo.domains.puddleworld` 2D Puddle world
================================================================

Puddle world is a 2D continuous world, populated with puddles of water.
The agent is required to safely navigate to a goal position while avoiding
these puddles. The penalty for entering the puddles is proportional to the
extent inside the puddle. Additionally, there is travel cost for each step
taken to encourage shorter paths.

.. automodule:: funzo.domains.puddleworld
.. autosummary::

    PuddleWorld
    PuddleWorldMDP
    PuddleReward
    PuddleRewardLFA
    PWTransition
    PWState


API
------

.. autoclass:: PuddleWorld
    :members:
.. autoclass:: PuddleWorldMDP
    :members:
.. autoclass:: PuddleReward
    :members:
.. autoclass:: PuddleRewardLFA
    :members:
.. autoclass:: PWTransition
    :members:
.. autoclass:: PWState
    :members:
.. autoclass:: PWAction
    :members:
