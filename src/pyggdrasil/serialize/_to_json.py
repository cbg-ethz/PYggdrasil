"""Serializes and deserializes a tree to JSON."""
import dataclasses
from typing import Any, Callable, Optional

from pyggdrasil.interface import MCMCSample
from pyggdrasil.tree import TreeNode, NameType, DataType
from pathlib import Path
import json
import xarray as xr

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


def save_mcmc_sample(sample: MCMCSample, output_dir: Path) -> None:
    """Appends MCMC sample to JSON file.

    Args:
        sample: MCMC sample to be saved
        output_dir: directory to save sample to

    Returns:
        None
    """

    sample_dict = sample.to_dict()
    # iteration = sample_dict["data_vars"]["iteration"]["data"]
    iteration = sample["iteration"].item()

    fullpath = output_dir / f"sample_{iteration}.json"

    with open(fullpath, "a") as f:
        json.dump(sample_dict, f)


def read_mcmc_samples(output_dir: Path, sample_id: int) -> list[MCMCSample]:
    """Reads in all MCMC samples from JSON file for a given run.

    Args:
        output_dir: directory to read sample from
        sample_id: sample number to read

    Returns:
        MCMC sample"""

    fullpath = output_dir / f"sample_{sample_id}.json"  # TODO: adjust to flexible title

    data = []
    with open(fullpath, "r") as f:
        for line in f:
            sample_dict = json.loads(line)
            data.append(xr.Dataset.from_dict(sample_dict))

    return data
