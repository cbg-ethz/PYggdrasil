"""Tests for serialization to JSON-compatible objects."""
from typing import Tuple

from pyggdrasil.tree import TreeNode

import pyggdrasil.serialize._to_json as tj

import pytest


def identity(x):
    """Identity function, used to (de)serialize data."""
    return x


@pytest.fixture
def tree_and_dict() -> Tuple[TreeNode, dict]:
    """Produces a tree and an equivalent dictionary."""
    root = TreeNode(name="root", data=1, parent=None)
    child1 = TreeNode(name="ch1", data=None, parent=root)
    TreeNode(name="ch2", data=2, parent=root)
    TreeNode(name="gr1", data=4, parent=child1)
    TreeNode(name="gr2", data=9, parent=child1)

    dictionary = {
        "name": "root",
        "data": 1,
        "children": [
            {
                "name": "ch1",
                "data": None,
                "children": [
                    {
                        "name": "gr1",
                        "data": 4,
                        "children": [],
                    },
                    {
                        "name": "gr2",
                        "data": 9,
                        "children": [],
                    },
                ],
            },
            {
                "name": "ch2",
                "data": 2,
                "children": [],
            },
        ],
    }

    return root, dictionary


def test_serialize_to_dict(tree_and_dict) -> None:
    """Serialization to dictionary."""
    root, expected_dict = tree_and_dict

    obtained = tj.serialize_tree_to_dict(root, serialize_data=identity)
    assert expected_dict == obtained


def test_blah(tree_and_dict) -> None:
    """Deserialization from dictionary."""
    root, dictionary = tree_and_dict
    tree_new = tj.deserialize_tree_from_dict(dictionary, deserialize_data=identity)

    assert str(tree_new) == str(root)
