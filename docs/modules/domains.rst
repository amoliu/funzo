:mod:`funzo.domains`
====================================

Domains define the world in which our reinforcement learning agents live. They
are implemented with the aim of hiding away all non-algorithmic details of such
environments so as to allow for easier switching of high level algorithms.

.. automodule:: funzo.domains
.. autosummary::

    Domain
    model_domain
    discretize_space
    distance_to_segment
    edist


Implemented domains
--------------------
.. toctree::
  :maxdepth: 1

  domains/gridworld
  domains/puddleworld
  domains/social_navigation



API
------
.. autoclass:: Domain
    :members:
.. autofunction:: model_domain

.. autofunction:: discretize_space
.. autofunction:: distance_to_segment
.. autofunction:: edist
