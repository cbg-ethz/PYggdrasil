"""Tree utilities."""
import anytree


def compare_trees(tree1: anytree.Node, tree2: anytree.Node) -> bool:
    """Compares two labeled rooted trees.

    Returns:
        True iff the two labeled rooted trees have the same structure

    Note:
        Assumes that labels in each tree is unique
    """
    # If the names or number of children differs, the trees are different
    if tree1.name != tree2.name or len(tree1.children) != len(tree2.children):
        return False

    # Sort children by their name to compare without order assumption
    sorted_children1 = sorted(tree1.children, key=lambda x: x.name)
    sorted_children2 = sorted(tree2.children, key=lambda x: x.name)

    # Now we recursively compare subtrees of matched children
    for child1, child2 in zip(sorted_children1, sorted_children2):
        if not compare_trees(child1, child2):
            return False

    return True
