:mod:`funzo.models`
====================================

.. automodule:: funzo.models

This module contains the design of key models used throughout the package. The main purpose of having models explicitly defined to to have clear interfaces and contracts on which algorithms can then be implemented. These include interfaces for :class:`MDP` and :class:`Domain` among others.

.. toctree::
    :hidden:

    models/MDP
    models/MDPState
    models/MDPAction
    models/RewardFunction
    models/TabularRewardFunction
    models/LinearRewardFunction


.. rubric:: :doc:`models/MDP`

.. autosummary::
    :nosignatures:

    R
    T
    S
    A
    actions
    terminal
    reward
    gamma
