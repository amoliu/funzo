====================================
Models (:mod:`pylfd.models`)
====================================
.. currentmodule:: pylfd.models

This module contains the design of key models used throughout the package. The main purpose of having models explicitely defined to to have clear interfaces and contracts on which algorithms can then be implemented. These include interfaces for :class:`MDP` and :class:`Domain` among others.

MDP
======

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~mdp.MDP
   ~mdp.MDPReward
   ~mdp.MDPRewardLFA
   ~mdp.MDPTransition
   ~mdp.MDPState
   ~mdp.MDPAction

Domain
===========

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~domain.Domain

