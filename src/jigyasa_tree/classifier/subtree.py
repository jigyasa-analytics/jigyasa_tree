from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn import tree
import copy
import numpy as np
import pandas as pd

def collect_nodes(tree):

    nodes=set()

    def dfs(node):
        if node==-1:
            return
        nodes.add(node)

        dfs(tree.children_left[node])
        dfs(tree.children_right[node])

    dfs(0)

    return nodes

def get_free_nodes(org_tree, mod_tree):
    org_nodes=collect_nodes(org_tree.tree_)
    mod_nodes=collect_nodes(mod_tree.tree_)

    removed= org_nodes - mod_nodes

    return removed

def create_manual_subtree(treeZ,node_index_to_changeZ,added_varZ,suggestions, tree_model_org):

    if suggestions is None:
        raise ValueError(f"suggestions cannot be None")

    feature_dict=suggestions.set_index("Feature Name").to_dict(orient="index")
    info=feature_dict[added_varZ]

    free_nodes=list(get_free_nodes(tree_model_org,treeZ))

    tree_ind = treeZ.tree_
    org_tree_ind=tree_model_org.tree_
    

    feature=added_varZ

    print(f"feature:{feature}")
    threshold=float(info["Threshold"])
    root_resp0=float(info["Root Node Resp=0"])
    root_resp1=float(info["Root Node Resp=1"])
    root_samples=int(info["Samples at root node"])
    root_entropy=float(info["Root Entropy"])

    feature_index=list(treeZ.feature_names_in_).index(feature)
    
    tree_ind.feature[node_index_to_changeZ]=feature_index
    tree_ind.value[node_index_to_changeZ]=[[root_resp0,root_resp1]]
    tree_ind.threshold[node_index_to_changeZ]=threshold
    tree_ind.impurity[node_index_to_changeZ]=root_entropy
    tree_ind.n_node_samples[node_index_to_changeZ]=root_samples


    left_id=tree_ind.children_left[node_index_to_changeZ]
    right_id=tree_ind.children_right[node_index_to_changeZ]

    left_id_org=org_tree_ind.children_left[node_index_to_changeZ]
    right_id_org=org_tree_ind.children_right[node_index_to_changeZ]

    left_child_ind=left_id_org
    
    print(f" {left_child_ind} is in {free_nodes}")     
    if left_child_ind in free_nodes and left_id_org !=-1:
        
        print("LEFT CHILD UPDATE")

        tree_ind.children_left[node_index_to_changeZ]=left_child_ind
        left_id=tree_ind.children_left[node_index_to_changeZ]
        
        left_samples=int(info["left Node Samples"])
        left_resp0=float(info["left Node resp= 0 "])
        left_resp1=float(info["left Node resp= 1 "])
        left_entropy=float(info["left Node Entropy"])

        tree_ind.value[left_id]=[[left_resp0,left_resp1]]
        tree_ind.impurity[left_id]=left_entropy
        tree_ind.n_node_samples[left_id]=left_samples

        tree_ind.children_left[left_id]=-1
        tree_ind.children_right[left_id]=-1
        tree_ind.feature[left_id]=-2
        tree_ind.threshold[left_id]=-2


    right_child_ind=right_id_org

    print(f" {right_child_ind} is in {free_nodes}")     
    if right_child_ind in free_nodes and right_id_org != -1:


        print("RIGHT CHILD UPDATE")

        tree_ind.children_right[node_index_to_changeZ]= right_child_ind
        right_id=tree_ind.children_right[node_index_to_changeZ]
        
        right_samples=int(info["Right Node Samples"])
        right_resp0=float(info["Right Node resp= 0 "])
        right_resp1=float(info["Right Node resp= 1 "])
        right_entropy=float(info["Right Node Entropy"])

        tree_ind.value[right_id]=[[right_resp0,right_resp1]]
        tree_ind.impurity[right_id]=right_entropy
        tree_ind.n_node_samples[right_id]=right_samples    


        tree_ind.children_left[right_id]=-1
        tree_ind.children_right[right_id]=-1
        tree_ind.feature[right_id]=-2
        tree_ind.threshold[right_id]=-2        


    if left_id_org == -1 and right_id_org ==-1:
        raise ValueError(f"  Node:{node_index_to_changeZ} is Leaf node. No feature to change. ")


    return treeZ

            

