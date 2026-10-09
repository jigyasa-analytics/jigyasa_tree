from sklearn.tree import _tree
import copy

def prune_node(model, node_id):
    
    pruned_model = copy.deepcopy(model)
    tree = pruned_model.tree_

    if tree.children_left[node_id] == -1  and  tree.children_right[node_id]== -1:
        raise ValueError("cannot prune LEAF node")

    def recurse(node):
        if node == -1:
            return

        if node == node_id:
            tree.children_left[node] = -1
            tree.children_right[node] = -1
            tree.feature[node] = _tree.TREE_UNDEFINED
            return

        recurse(tree.children_left[node])
        recurse(tree.children_right[node])

    recurse(0)
    print(f"Tree is pruned at NODE{node_id}")
    return pruned_model

            

