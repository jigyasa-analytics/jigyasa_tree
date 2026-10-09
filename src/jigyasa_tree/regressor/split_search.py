from sklearn.tree import DecisionTreeRegressor, _tree
from sklearn import tree
import copy
import numpy as np
import pandas as pd

def calculate_mse(y):
    mse =np.mean((y - np.mean(y)**2))
    return mse

def get_node_counts(tree_structure, node_id):
   value=tree_structure.value[node_id].flatten()
   samples= tree_structure.n_node_samples[node_id]
   if np.isclose(value.sum(),1.0):
      value=value * samples

   return value

def suggest_splits(datasetZ, DEPVAR, Var_listZ, prune_node):

    col=f"NO{prune_node}"

    if col not in datasetZ.columns:
        raise ValueError(f"{col} does not exist")

    datasetZ_filtered = datasetZ[datasetZ[col]==1].copy()
    
    

    Y = datasetZ_filtered[DEPVAR]
    X = datasetZ_filtered[Var_listZ]

    num_samples, num_features = np.shape(X)
    print(X.head())
    print("num_samples and num_features")
    print(num_samples)
    print(num_features)
        
    best_split = {}
    #max_info_gain = -float("inf")
    
    df_list=[]
    
    for cols in X.columns:
        feature_names=X.columns

        clf=DecisionTreeRegressor(criterion="squared_error", max_depth=1, min_samples_leaf = 0.10)
        clf=clf.fit(X , Y)
        

        tree_rules0 = tree.export_graphviz(clf,feature_names = X.columns)
        print(tree_rules0)

        feature_array=np.array(clf.tree_.feature)
        if feature_array[0]!=-2:
            srlno_feature = feature_array[0]
        else:
            srlno_feature=0
        print("Serial_no_feature:", srlno_feature)
            
             
        drop_feature = X.columns[srlno_feature]  
        print("drop_feature:", drop_feature)

        print("X_before_Drop", X.head())
            
        X = X.drop([drop_feature], axis = 1)
        print("chk X", X)
        ###############################################################################################################

        tree_structure=clf.tree_            

        root_node_id=0
        root_feature=feature_names[tree_structure.feature[root_node_id]] if tree_structure.feature[root_node_id]!=-2 else 'Leaf'
        root_threshold=tree_structure.threshold[root_node_id] if tree_structure.feature[root_node_id]!=-2 else None
        root_impurity=tree_structure.impurity[root_node_id]
        root_samples=tree_structure.n_node_samples[root_node_id]
        root_value=get_node_counts(tree_structure, root_node_id)
        root_mean_target_value=tree_structure.value[root_node_id].flatten()[0]
        root_mse = root_impurity
        root_rmse = np.sqrt(root_impurity)

        leaf_nodes1={'left Node': 1,'Right Node': 2}
        leaf_nodes=[1,2]
        
        leaf_details={}
        print("ROOT_FEATURE", root_feature)

        if root_feature == "Leaf":
            continue

        for position,node_id in leaf_nodes1.items():
                leaf_value0_chk=tree_structure.value[node_id]
                print("leaf_value0_chk", leaf_value0_chk)
                leaf_value=leaf_value0_chk.flatten()
                
                feature=data.feature_names[tree_structure.feature[node_id]] if tree_structure.feature[node_id]!=-2 else 'Leaf'
                threshold=tree_structure.threshold[node_id] if tree_structure.feature[node_id]!=-2 else None
                impurity=tree_structure.impurity[node_id]
                samples=tree_structure.n_node_samples[node_id]
                value=tree_structure.value[node_id].flatten()              
                leaf_mean_target=tree_structure.value[node_id].flatten()[0]
                leaf_mse=tree_structure.impurity[node_id]      
             

                leaf_details[position]={
                    'Leaf Node ID':node_id,
                    'Leaf Feature':feature,
                    'Leaf Threshold':threshold,
                    'Leaf Impurity':impurity,
                    'Leaf Samples':samples,
                    'Leaf Value':value,
                    'Mean Target':leaf_mean_target,
                    'MSE':leaf_mse 
                }

                combined_row={
                    'Feature Name':root_feature,
                    'Threshold':root_threshold,
                    'Root Impurity':root_impurity,
                    'Samples at root node':root_samples,
                    'Root Node Mean Target':root_mean_target_value
                }


        for position,details in leaf_details.items() :
                combined_row[f'{position} Samples']=details['Leaf Samples']
                combined_row[f'{position} Mean Target']=details['Mean Target']
                combined_row[f'{position} MSE']=details['MSE']


        df=pd.DataFrame([combined_row])
        print(df)



        total_samples=np.sum(df['Samples at root node'])
        total_samples=np.sum(df['Samples at root node'])
        df['Weighted_Impurity']=(df['left Node Samples']/total_samples)*df['left Node MSE'] + (df['Right Node Samples']/total_samples)*df['Right Node MSE']
        df['Impurity_Reduction']=(df['Root Impurity']-df['Weighted_Impurity'])


        print(df)

        df_list.append(df)
            

    combined_df=pd.concat(df_list,ignore_index=True)

    combined_df_cleaned=combined_df.drop_duplicates()

    df_sorted=combined_df_cleaned.sort_values(by='Impurity_Reduction',ascending=False)
    df_sorted1 = df_sorted.rename(columns={"Root Impurity":"Root MSE"})
 
  
    best_split["feature"] = df_sorted["Feature Name"].iloc[0]

    print("best_split")
    print(best_split)
    
    return best_split,df_sorted1  


            

