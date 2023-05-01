"""Tree utilities."""
import anytree


def compare_trees(tree1: anytree.Node, tree2: anytree.Node) -> bool:
    """Compares two trees."""
    if tree1.name != tree2.name:
        return False

    if len(tree1.children) != len(tree2.children):
        return False

    # Sort children by their tag to compare without order assumption
    sorted_children1 = sorted(tree1.children, key=lambda x: x.name)
    sorted_children2 = sorted(tree2.children, key=lambda x: x.name)

    for child1, child2 in zip(sorted_children1, sorted_children2):
        if not compare_trees(child1, child2):
            return False

    return True
