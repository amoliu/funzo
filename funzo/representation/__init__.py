"""
The :mod:`funzo.representation` module contains various way of representing
the models used in the library. Simple tabular representation is assumed.
These could be graphs, neural networks, etc
"""


from .state_graph import StateGraph


__all__ = [
    'StateGraph',
]
