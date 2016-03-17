====================================
Models (:mod:`funzo.models`)
====================================
.. currentmodule:: funzo.models

This module contains the design of key models used throughout the package. The main purpose of having models explicitely defined to to have clear interfaces and contracts on which algorithms can then be implemented. These include interfaces for :class:`MDP` and :class:`Domain` among others.

MDP
======
Markov decision processes

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~mdp.MDP
   ~mdp.RewardFunction
   ~mdp.TabularRewardFunction
   ~mdp.LinearRewardFunction
   ~mdp.MDPTransition
   ~mdp.MDPState
   ~mdp.MDPAction


StateGraph
===========
Generic graph based representation

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~state_graph.StateGraph
