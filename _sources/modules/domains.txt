:mod:`funzo.domains`
====================================

.. automodule:: funzo.domains

.. autosummary::

    Domain
    model_domain

1. Grid World

    .. autosummary::

        GridWorld
        GridWorldMDP
        GReward
        GRewardLFA
        GTransition
        GState
        GAction


2. Puddle World

    .. autosummary::

        PuddleWorld
        PuddleWorldMDP
        PuddleReward
        PuddleRewardLFA
        PWTransition
        PWState
        PWAction

.. autosummary::

    discretize_space
    distance_to_segment
    edist


Detailed descriptions
-----------------------
.. autoclass:: Domain
    :memebers:
.. autofunction:: model_domain

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

.. autofunction:: discretize_space
.. autofunction:: distance_to_segment
.. autofunction:: edist
