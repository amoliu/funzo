====================================
Planners (:mod:`funzo.planners`)
====================================
.. currentmodule:: funzo.planners

This module contains implementation of various planning algorithms for Markov decision processes in general. Such planners take in MDPs and additional relevant parameters and compute policies and/or value functions either using exact methods or by approximation.

Exact Planning
=================
The set of planners that work exhaustively on the the state and action space of MDPs (usually discrete and small or medium sized).

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~exact.policy_iteration
   ~exact.value_iteration

