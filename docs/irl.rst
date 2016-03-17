==============================================================
Inverse Reinforcement Learning (IRL) (:mod:`funzo.irl`)
==============================================================
.. currentmodule:: funzo.irl

This module contains the various IRL algorithms implemented in the library.

Bayesian IRL
================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~irl.BIRL
   ~irl.birl.UniformRewardPrior
   ~irl.birl.LaplacianRewardPrior
   ~irl.birl.GaussianRewardPrior
   ~irl.birl.DirectionalRewardPrior
   ~irl.birl.mcmc_birl.PolicyWalkBIRL
   ~irl.birl.mcmc_birl.PolicyWalkProposal
   ~irl.birl.map_birl.MAPBIRL
