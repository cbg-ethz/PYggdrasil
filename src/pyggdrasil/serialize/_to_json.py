"""Serializes and deserializes a tree to JSON."""
import dataclasses
from typing import Any, Callable, Optional

from pyggdrasil.tree import TreeNode, NameType, DataType

DictSeralizedFormat = dict


@dataclasses.dataclass
class _NamingConvention:
    NAME: str = "name"
    DATA: str = "data"
    CHILDREN: str = "children"


def serialize_tree_to_dict(
    tree_root: TreeNode[NameType, DataType],
    *,
    serialize_data: Callable[[DataType], Any],
    naming: Optional[_NamingConvention] = None,
) -> DictSeralizedFormat:
    """Serializes a tree with data into a nested dictionary.

    Args:
        tree_root: node representing the root of the tree to be serialized
        serialize_data: function serializing a ``DataType`` object
        naming: serialization naming conventions, non-default values
          are discouraged

    Returns:
        dictionary storing the tree in the format:
        {
            "name": "Root name",
            "data": [serialized root.data],
            "children": [
                {
                    "name": "Child name",
                    "data": [serialized child.data],
                    "children": [
                        ...
                    ]
                }
                ...
            ]
        }
    """
    naming = naming or _NamingConvention()

    return {
        naming.NAME: tree_root.name,
        naming.DATA: serialize_data(tree_root.data),
        naming.CHILDREN: [
            serialize_tree_to_dict(child, serialize_data=serialize_data)
            for child in tree_root.children
        ],
    }


def deserialize_tree_from_dict(
    dct: DictSeralizedFormat,
    *,
    deserialize_data: Callable[[Any], DataType],
    naming: Optional[_NamingConvention] = None,
) -> TreeNode:
    """Creates tree from dictionary in the ``DictSerializedFormat``.

    Args:
        dct: dictionary to be read
        deserialize_data: factory method creating ``DataType`` objects from dictionaries
        naming: serialization naming conventions, non-default values
          are discouraged

    Returns:
        root node to the generated tree
    """
    naming = naming or _NamingConvention()

    def generate_node(node_dict: dict, parent: Optional[TreeNode]) -> TreeNode:
        """Auxiliary method building node from ``node_dict``
        and connecting it to ``parent``."""
        new_node = TreeNode(
            name=node_dict[naming.NAME],
            data=deserialize_data(node_dict[naming.DATA]),
            parent=parent,
        )

        for child_dict in node_dict[naming.CHILDREN]:
            generate_node(node_dict=child_dict, parent=new_node)

        return new_node

    return generate_node(dct, parent=None)
