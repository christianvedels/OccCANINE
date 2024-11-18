from collections import defaultdict


class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.codes = []

    def count_end_nodes(self) -> int:
        """
        Including for ease of debugging. Number of end nodes should always be equal to the len(codes_list)
        Count the total number of end nodes (leaf nodes) in the trie.
        """
        if not self.children:  # If there are no children, this is a leaf node
            return 1

        count = 0

        for child in self.children.values():
            count += child.count_end_nodes()  # Recursively count the end nodes

        return count

    def count_nodes(self) -> int:
        """
        Count the total number of nodes in the trie.
        """
        count = 1  # Count the current node

        for child in self.children.values():
            count += child.count_nodes()  # Recursively count the children

        return count


def build_trie(codes_list):
    root = TrieNode()

    for code in codes_list:
        node = root

        for number in code:
            node = node.children[number]

        node.codes.append(code)

    return root
