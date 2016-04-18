"""
The :mod:`funzo.utils` module contains general helper functions and classes
"""

from .data_structures import Trace

from .validation import check_random_state


__all__ = [
    'Trace',
    #
    'check_random_state',
    #
]
