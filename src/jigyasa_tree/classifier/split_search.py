from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn import tree
import copy
import numpy as np
import pandas as pd
import inspect
from sklearn import __version__ as sklearn_version
from packaging.version import Version

def use_proportions():
    return Version(sklearn_version) >= Version("1.4")


def calculate_entropy(samples): 
   total_samples=np.sum(samples)
   if total_samples <= 0:
       return 0.0
   proportions=samples/total_samples

   proportions=proportions[proportions>0]
   
   entropy=-np.sum(proportions*np.log2(proportions))
   return entropy


def calculate_response_rate(node_value):
    total_samples=node_value.sum()
    if total_samples <= 0:
       return np.zeros_like(node_value, dtype=float)
    response_rates=node_value/total_samples
    return response_rates


def get_node_counts(tree_structure, node_id):
   value=tree_structure.value[node_id].flatten().copy()
   
   if use_proportions():
      value=value * tree_structure.weighted_n_node_samples[node_id]

   return value

def get_all_classes_counts(tree_structure, node_id, model_classes, all_classes):
   value=get_node_counts(tree_structure, node_id)

   counts=np.zeros(len(all_classes), dtype=float)

   class_positions={cls: i for i, cls in enumerate(all_classes)}

   for i, cls in enumerate(model_classes):
      counts[class_positions[cls]]=value[i]

   return counts

def suggest_splits(datasetZ, RESPONSE_INDZ, Var_listZ, prune_node, all_classes, sample_weight=None, **split_params):

    # Only depth is fixed. Reuse effective original sample weights, so
    # class_weight must not be applied a second time to candidate trees.
    if "max_depth" in split_params:
        raise ValueError("max_depth is fixed at 1 for manual subtrees")
    if "class_weight" in split_params:
        raise ValueError("class_weight is inherited from the original fit; do not specify it again")
    allowed = set(inspect.signature(DecisionTreeClassifier).parameters) - {"max_depth", "class_weight"}
    unknown = set(split_params) - allowed
    if unknown:
        raise TypeError(f"Unsupported split parameters: {sorted(unknown)}")
    params = dict(criterion="entropy", min_samples_leaf=0.05, max_depth=1)
    params.update(split_params)
    # Candidate search removes features successively; constraints must follow
    # the remaining feature columns at each iteration.
    monotonic = params.pop("monotonic_cst", None)
    if monotonic is not None and len(monotonic) != len(Var_listZ):
        raise ValueError("monotonic_cst must match var_list length")

    col=f"NO{prune_node}"

    if col not in datasetZ.columns:
        raise ValueError(f"{col} does not exist")


    datasetZ_filtered = datasetZ[datasetZ[col]==1].copy()
    

    # Position-based alignment with the original fit, including duplicate DataFrame indices.
    weights = None
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)[datasetZ[col].to_numpy() == 1]

    Y = datasetZ_filtered[RESPONSE_INDZ]
    X = datasetZ_filtered[Var_listZ]



    num_samples, num_features = np.shape(X)
        
    best_split = {}
    #max_info_gain = -float("inf")
    
    df_list=[]
    
    for cols in X.columns:
        feature_names=X.columns

        candidate_params = params.copy()
        if monotonic is not None:
            candidate_params["monotonic_cst"] = [monotonic[list(Var_listZ).index(f)] for f in X.columns]
        clf=DecisionTreeClassifier(**candidate_params)
        clf=clf.fit(X , Y, sample_weight=weights)
        
        #Extract tree structure
        tree_rules0 = tree.export_graphviz(clf,feature_names = X.columns)

        feature_array=np.array(clf.tree_.feature)
        if feature_array[0]!=-2:
            srlno_feature = feature_array[0]
        else:
            srlno_feature=0
            
             
        drop_feature = X.columns[srlno_feature]  ## feature to drop in next round

            
        X = X.drop([drop_feature], axis = 1)
        ###############################################################################################################

        tree_structure=clf.tree_            

        root_node_id=0
        root_feature=feature_names[tree_structure.feature[root_node_id]] if tree_structure.feature[root_node_id]!=-2 else 'Leaf'
        root_threshold=tree_structure.threshold[root_node_id] if tree_structure.feature[root_node_id]!=-2 else None
        root_impurity=tree_structure.impurity[root_node_id]
        root_samples=tree_structure.n_node_samples[root_node_id]
        root_weight=tree_structure.weighted_n_node_samples[root_node_id]
        root_value=get_all_classes_counts(tree_structure, root_node_id, clf.classes_, all_classes)
        root_response_rate=calculate_response_rate(root_value)

        leaf_nodes1={'Left Node': 1,'Right Node': 2}
        leaf_nodes=[1,2]
        
        leaf_details={}

        if root_feature == "Leaf":
            continue

        for position,node_id in leaf_nodes1.items():
            leaf_value0_chk=tree_structure.value[node_id]
            leaf_value=leaf_value0_chk.flatten()
                
            feature=feature_names[tree_structure.feature[node_id]] if tree_structure.feature[node_id]!=-2 else 'Leaf'
            threshold=tree_structure.threshold[node_id] if tree_structure.feature[node_id]!=-2 else None
            impurity=tree_structure.impurity[node_id]
            samples=tree_structure.n_node_samples[node_id]
            weighted_samples=tree_structure.weighted_n_node_samples[node_id]
            value=get_all_classes_counts(tree_structure, node_id, clf.classes_, all_classes)
            leaf_response_rate=calculate_response_rate(value)
            leaf_impurity=impurity        
             

            leaf_details[position]={
                'Leaf Node ID':node_id,
                'Leaf Feature':feature,
                'Leaf Threshold':threshold,
                'Leaf Impurity':impurity,
                'Leaf Samples':samples,
                'Leaf Weighted Samples':weighted_samples,
                'Leaf Value':value,
                'Response Rate':leaf_response_rate,
                'Impurity':leaf_impurity
            }

        combined_row={
                'Feature Name':root_feature,
                'Criterion':candidate_params['criterion'],
                'Threshold':root_threshold,
                'Samples at root node':root_samples,
                'Weighted Samples at root node':root_weight,
                'Root Impurity':root_impurity
                }


        for i, cls in enumerate(all_classes):
           combined_row[f'Root Node resp={cls}']=root_response_rate[i]



        for position,details in leaf_details.items() :
            combined_row[f'{position} Samples']=details['Leaf Samples']
            combined_row[f'{position} Weighted Samples']=details['Leaf Weighted Samples']
            combined_row[f'{position} Impurity']=details['Impurity']                    

            
            for i, cls in enumerate(all_classes):
               combined_row[f'{position} resp={cls}']=details['Response Rate'][i]


        df=pd.DataFrame([combined_row])



        total_samples = df['Weighted Samples at root node']
        df['Impurity_Reduction']=df['Root Impurity']-(
            (df['Left Node Weighted Samples']/total_samples)*df['Left Node Impurity'] +
            (df['Right Node Weighted Samples']/total_samples)*df['Right Node Impurity']
            )


        df_list.append(df)
            

    if not df_list:
        raise ValueError(
            f"No valid split found at node {prune_node} for variables {list(Var_listZ)}. "
            "Try other variables (constant or near-constant columns cannot split the node)"
        )

    combined_df=pd.concat(df_list,ignore_index=True)


    combined_df_cleaned=combined_df.drop_duplicates()

    df_sorted=combined_df_cleaned.sort_values(by='Impurity_Reduction',ascending=False)       
 
  
    best_split["feature"] = df_sorted["Feature Name"].iloc[0]

    
    return best_split,df_sorted  


            

