====================================
Utils (:mod:`funzo.utils`)
====================================
.. currentmodule:: funzo.utils

This module contains a number of convenient utilities used in the package such as data structures, little cute computations, etc

Data Structures
=================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~data_structures.Trace


MCMC Diagnostics
=================
Set of functions for checking the properties of resulting traces from running MCMC algorithms such as PolicyWalk.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~diagnostics.geweke
   ~diagnostics.autocorr
   ~diagnostics.autocov


Random Numbers
=================
For consistent generation of random numbers, used throughout the library, a set of tools (some borrowed from scikit-learn project are used)

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~validation.check_random_state
