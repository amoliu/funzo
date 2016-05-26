.. image:: https://img.shields.io/travis/makokal/funzo.svg
        :target: https://travis-ci.org/makokal/funzo

.. image:: https://codecov.io/gh/makokal/funzo/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/makokal/funzo

.. image:: https://requires.io/github/makokal/funzo/requirements.svg?branch=master
     :target: https://requires.io/github/makokal/funzo/requirements/?branch=master
     :alt: Requirements Status

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/makokal/funzo/blob/master/LICENSE

funzo
============
Python toolkit for (inverse) reinforcement learning (IRL). This toolkit strives for flexible interfaces enabling generic implementations of standard algorithms for RL and IRL.

Documentation: `http://makokal.github.io/funzo/ <http://makokal.github.io/funzo/>`__

Features
---------
* Flexible interfaces for models and algorithms
* Inverse reinforcement learning algorithms
    - Bayesian IRL
        - PolicyWalk based BIRL
        - MAP based BIRL
* RL domains for experiments
    - GridWorld (discrete)
    - Chainworld (discrete)
    - PuddleWorld (continuous)
    - Social Navigation (continuous)
* MDP solvers
    - Policy Iteration
    - Value Iteration


Usage
------------
See `examples <examples>`_ folder.


What about the name?
----------------------
**funzo** is a Swahili word for instruction or doctrine or simply teaching, which is what we try to achieve here using IRL. If we do not succeed, the end result may not be desirable as Homer Simpson found out with `Homers funzo <http://simpsons.wikia.com/wiki/Funzo>`__
