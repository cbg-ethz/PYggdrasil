"""This submodule defines our data format for trees, based
on AnyTree.
"""
from typing import Generic, Iterable, Optional, TypeVar

import anytree

NameType = TypeVar("NameType", int, str)
DataType = TypeVar("DataType")


class TreeNode(Generic[NameType, DataType], anytree.NodeMixin):
    """Tree node compatible with AnyTree.
    It can be annotated with the label type and the data type.

    Attrs:
        name: identifier of the node (compatible with declared type)
        data: payload data
        parent: the node of the parent
        children: iterable with the children nodes
    """

    def __init__(
        self,
        name: NameType,
        data: DataType = None,
        parent: Optional["TreeNode"] = None,
        children: Optional[Iterable["TreeNode"]] = None,
    ) -> None:
        """

        Args:
            name: identifier of the node (compatible with declared type)
            data: payload data
            parent: the node of the parent
            children: iterable with the children nodes
        """
        self.name = name
        self.data = data

        self.parent = parent
        if children:
            self.children = children

    def __str__(self) -> str:
        """Casts to str."""
        ret = ""
        for pre, _, node in anytree.RenderTree(self):
            ret += f"{pre}{node.name}: {str(node.data)}\n"
        return ret

    def print_topo(self) -> None:
        """
        Prints the topology of the tree.
        """
        for pre, _, node in anytree.RenderTree(self):
            print("%s%s" % (pre, node.name))

    @staticmethod
    def convert_anytree_to_treenode(node: anytree.Node) -> "TreeNode":
        """Converts an AnyTree node to a TreeNode.

        Ignores the data attribute TreeNode nodes.

        Args:
              node: AnyTree node to be converted
        Returns:
            TreeNode with the same topology as the AnyTree node
        """
        children = [
            TreeNode.convert_anytree_to_treenode(child) for child in node.children
        ]
        return TreeNode(name=node.name, children=children)


__all__ = ["TreeNode", "DataType", "NameType"]
