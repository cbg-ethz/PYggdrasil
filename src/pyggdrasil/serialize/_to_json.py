"""Serializes and deserializes a tree to JSON."""
import dataclasses
import jax
import numpy as np
from typing import Any, Callable, Optional, Union

from pyggdrasil.interface import MCMCSample
from pyggdrasil.tree import TreeNode, NameType, DataType


from pathlib import Path
import json
import xarray as xr
import jax.numpy as jnp

DictSeralizedFormat = dict

# Type annotation for a generic array.
Array = Union[jax.Array, np.ndarray]


class JnpEncoder(json.JSONEncoder):
    """Encoder for numpy types."""

    def default(self, obj):
        """Default encoder."""
        if isinstance(obj, jnp.integer):
            return int(obj)
        if isinstance(obj, jnp.floating):
            # 👇️ alternatively use str()
            return float(obj)
        if isinstance(obj, Array):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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
    deserialize_data: Callable[[Any], DataType],  # type: ignore
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


def save_tree_node(tree: TreeNode, output_fp: Path):
    """Saves Tree object as dict /json to disk.

    Args:
        tree: Tree object to be saved
        output_fp: directory to save tree to
    Returns:
        None
    """

    tree_node = serialize_tree_to_dict(tree, serialize_data=lambda x: x)

    # make path
    output_fp = Path(output_fp)

    # create directory if it doesn't exist
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    # make file if it doesn't exist
    output_fp.touch(exist_ok=True)

    with open(output_fp, "w") as f:
        json.dump(tree_node, f, cls=JnpEncoder)


def read_tree_node(fp: Path) -> TreeNode:
    """Reads Json file to Tree object from disk.

    Args:
        fp: directory to save tree to

    """

    with open(fp, "r") as f:
        tree_node = json.load(f)

    return deserialize_tree_from_dict(tree_node, deserialize_data=lambda x: x)


def save_mcmc_sample(sample: MCMCSample, out_fp: Path) -> None:
    """Appends MCMC sample to JSON file.

    Args:
        sample: MCMC sample to be saved
        out_fp: path to JSON file

    Returns:
        None
    """

    sample_dict = sample.to_dict()

    fullpath = out_fp

    with open(fullpath, "a") as f:
        json_str = json.dumps(sample_dict)
        f.write(json_str + "\n")


def read_mcmc_samples(fullpath: Path) -> list[MCMCSample]:
    """Reads in all MCMC samples from JSON file for a given run.

    Args:
        fullpath: path to JSON file

    Returns:
        MCMC sample"""

    data = []
    with open(fullpath, "r") as f:
        for line in f:
            sample_dict = json.loads(line)
            data.append(xr.Dataset.from_dict(sample_dict))

    return data


def save_metric_result(
    axis: list[Union[int, str]],
    result: list[float],
    out_fp: Path,
    axis_name: str = "iteration",
) -> None:
    """Appends metric result to JSON file."""
    # make dict
    metric_dict = {axis_name: axis, "result": result}
    # make path
    out_fp = Path(out_fp)
    # create directory if it doesn't exist
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    # make file if it doesn't exist
    out_fp.touch(exist_ok=True)
    # write to file
    with open(out_fp, "w") as f:
        json_str = json.dumps(metric_dict, indent=4, cls=JnpEncoder)
        f.write(json_str + "\n")


def read_metric_result(fullpath: Path) -> tuple[list[Union[int, str]], list[float]]:
    """Reads in all metric results from JSON file for a given run."""
    with open(fullpath, "r") as f:
        metric_dict = json.load(f)

    # get axis name
    axis_name = list(metric_dict.keys())[0]

    # convert metric result to list of floats
    metric_dict["result"] = [float(x) for x in metric_dict["result"]]

    return metric_dict[axis_name], metric_dict["result"]
