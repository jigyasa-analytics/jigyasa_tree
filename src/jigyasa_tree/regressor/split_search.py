from sklearn.tree import DecisionTreeRegressor, _tree
from sklearn import tree
import copy
import numpy as np
import pandas as pd
import inspect
from sklearn import __version__ as sklearn_version
from packaging.version import Version

def use_proportions():
    return Version(sklearn_version) >= Version("1.4")

def calculate_mse(y):
    mse = np.mean((y - np.mean(y)) ** 2)
    return mse

def get_node_counts(tree_structure, node_id):
   value=tree_structure.value[node_id].flatten()
   samples= tree_structure.n_node_samples[node_id]
   if use_proportions():
      value=value * samples

   return value

def suggest_splits(datasetZ, DEPVAR, Var_listZ, prune_node, sample_weight=None, **split_params):

    # Fixed depth; criterion inherited from the original model.
    if "max_depth" in split_params:
        raise ValueError("max_depth is fixed at 1 for manual subtrees")
    allowed = set(inspect.signature(DecisionTreeRegressor).parameters) - {"max_depth"}
    unknown = set(split_params) - allowed
    if unknown:
        raise TypeError(f"Unsupported split parameters: {sorted(unknown)}")
    params = dict(criterion="squared_error", min_samples_leaf=0.05, max_depth=1)
    params.update(split_params)
    selected_criterion = params["criterion"]
    # Candidate search removes features successively; constraints must follow
    # the remaining feature columns at each iteration.
    monotonic = params.pop("monotonic_cst", None)
    if monotonic is not None and len(monotonic) != len(Var_listZ):
        raise ValueError("monotonic_cst must match var_list length")

    col=f"NO{prune_node}"

    if col not in datasetZ.columns:
        raise ValueError(f"{col} does not exist")

    mask = datasetZ[col].to_numpy() == 1
    datasetZ_filtered = datasetZ.iloc[np.flatnonzero(mask)].copy()
    node_weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)[mask]
    
    

    Y = datasetZ_filtered[DEPVAR]
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
        X_before_split = X.copy()
        clf=DecisionTreeRegressor(**candidate_params)
        clf=clf.fit(X , Y, sample_weight=node_weights)
        

        tree_rules0 = tree.export_graphviz(clf,feature_names = X.columns)

        feature_array=np.array(clf.tree_.feature)
        if feature_array[0] == _tree.TREE_UNDEFINED:
            break

        srlno_feature = feature_array[0]            
             
        drop_feature = X.columns[srlno_feature]  

            
        X = X.drop([drop_feature], axis = 1)
        ###############################################################################################################

        tree_structure=clf.tree_            

        root_node_id=0
        root_feature=feature_names[tree_structure.feature[root_node_id]] if tree_structure.feature[root_node_id]!=-2 else 'Leaf'
        root_threshold=tree_structure.threshold[root_node_id] if tree_structure.feature[root_node_id]!=-2 else None
        root_impurity=tree_structure.impurity[root_node_id]
        root_samples=tree_structure.n_node_samples[root_node_id]
        root_value=get_node_counts(tree_structure, root_node_id)
        root_mean_target_value=tree_structure.value[root_node_id].flatten()[0]

        leaf_nodes1={'left Node': 1,'Right Node': 2}
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
                value=tree_structure.value[node_id].flatten()              
                leaf_mean_target=tree_structure.value[node_id].flatten()[0]
     
             

                leaf_details[position]={
                    'Leaf Node ID':node_id,
                    'Leaf Feature':feature,
                    'Leaf Threshold':threshold,
                    'Leaf Impurity':impurity,
                    'Leaf Samples':samples,
                    'Weighted Samples':tree_structure.weighted_n_node_samples[node_id],
                    'Leaf Value':value,
                    'Mean Target':leaf_mean_target,
                }

                combined_row={
                    'Feature Name':root_feature,
                    'Threshold':root_threshold,
                    'Root Impurity':root_impurity,
                    'Samples at root node':root_samples,
                    'Root Weighted Samples':tree_structure.weighted_n_node_samples[0],
                    'Root Node Mean Target':root_mean_target_value
                }


        for position,details in leaf_details.items() :
                combined_row[f'{position} Samples']=details['Leaf Samples']
                combined_row[f'{position} Weighted Samples']=details['Weighted Samples']
                combined_row[f'{position} Mean Target']=details['Mean Target']
                combined_row[f'{position} Criterion Impurity']=details['Leaf Impurity']


        df=pd.DataFrame([combined_row])

        df['Criterion'] = selected_criterion
        df['Root Criterion Impurity'] = root_impurity
        # Independently compute actual (weighted) MSE/RMSE for all criteria.
        y_values = Y.to_numpy(dtype=float)
        effective_weights = np.ones(len(y_values)) if node_weights is None else node_weights
        def actual_mse(node_mask):
            y_node = y_values[node_mask]
            w_node = effective_weights[node_mask]
            mean = np.average(y_node, weights=w_node)
            return float(np.average((y_node - mean)**2, weights=w_node))
        left_mask = X_before_split[feature_names[tree_structure.feature[0]]].to_numpy() <= root_threshold
        df['Root Actual MSE'] = actual_mse(np.ones(len(Y), dtype=bool))
        df['left Node Actual MSE'] = actual_mse(left_mask)
        df['Right Node Actual MSE'] = actual_mse(~left_mask)
        for prefix in ('Root', 'left Node', 'Right Node'):
            df[f'{prefix} Actual RMSE'] = np.sqrt(df[f'{prefix} Actual MSE'])
        df['Weighted_Impurity']=(df['left Node Weighted Samples']/df['Root Weighted Samples'])*df['left Node Criterion Impurity'] + (df['Right Node Weighted Samples']/df['Root Weighted Samples'])*df['Right Node Criterion Impurity']
        df['Impurity_Reduction']=(df['Root Impurity']-df['Weighted_Impurity'])



        df_list.append(df)
            

    if not df_list:
        raise ValueError(
            f"No valid split found at node {prune_node} for variables {list(Var_listZ)}. "
            "Try other variables (constant or near-constant columns cannot split the node)"
        )

    combined_df=pd.concat(df_list,ignore_index=True)

    combined_df_cleaned=combined_df.drop_duplicates()

    df_sorted=combined_df_cleaned.sort_values(by='Impurity_Reduction',ascending=False)
    df_sorted1 = df_sorted.drop(columns=["Root Impurity"])
 
  
    best_split["feature"] = df_sorted["Feature Name"].iloc[0]

    
    return best_split,df_sorted1  


            

