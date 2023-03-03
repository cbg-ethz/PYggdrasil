"""Serializes and deserializes a tree to JSON."""
from typing import Callable

from pyggdrasil.tree import TreeNode, NameType, DataType

DictSeralizedFormat = dict


def serialize_tree_to_dict(
    tree_root: TreeNode[NameType, DataType],
    data_to_dict: Callable[[DataType], dict],
) -> DictSeralizedFormat:
    """Serializes a tree with data into a nested dictionary.

    Args:
        tree_root: node representing the root of the tree to be serialized
        data_to_dict: function serializing a ``DataType`` object into a dictionary

    Returns:
        dictionary storing the tree
    """
    raise NotImplementedError


def deserialize_tree_from_dict(
    dct: DictSeralizedFormat,
    dict_to_data: Callable[[dict], DataType],
) -> TreeNode:
    """Creates tree from dictionary in the ``DictSerializedFormat``.

    Args:
        dct: dictionary to be read
        dict_to_data: factory method creating ``DataType`` objects from dictionaries

    Returns:
        root node to the generated tree
    """

    raise NotImplementedError
