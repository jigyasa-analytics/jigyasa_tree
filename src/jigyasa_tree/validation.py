
import pandas as pd
import numpy as np

def check_training_data(X_train, y_train):
    

    if X_train.empty:
        raise ValueError("Training Data is Empty")


    if len(X_train) != len(y_train):
        raise ValueError("X and y Must have same number of rows")


    if X_train.isna().any().any():
        raise ValueError("Input features containing missing values")

    if y_train.isna().any():
        raise ValueError("Target containing missing values")


    if y_train.nunique() < 2:
        raise ValueError("Target contains only one class")
    

    if len(X_train.columns) != len(set(X_train.columns)):
        raise ValueError("Feature Names must be unique")

    if np.issubdtype(y_train, np.floating):
        if not np.all(np.mod(y_train, 1) == 0):
            raise ValueError("Continuous targets detected. Regression is not supported")

    if y_train.nunique() > 2:
        raise ValueError("Only binary classification is supported.Target contains more than two classes")

    


def check_val_data(X_val, y_val):
    

    if X_val.empty:
        raise ValueError("Validation Data is Empty")


    if len(X_val) != len(y_val):
        raise ValueError("X_val and y_val Must have same number of rows")


    if X_val.isna().any().any():
        raise ValueError("Input features containing missing values")

    if y_val.isna().any():
        raise ValueError("Target containing missing values")


    if y_val.nunique() < 2:
        raise ValueError("Traget contains only one class")
    

    if len(X_val.columns) != len(set(X_val.columns)):
        raise ValueError("Feature Names must be unique")


    if np.issubdtype(y_val, np.floating):
        if not np.all(np.mod(y_val, 1) == 0):
            raise ValueError("Continuous targets detected. Regression is not supported")


    if y_val.nunique() > 2:
        raise ValueError("Only binary classification is supported.Target contains more than two classes")

    
            


def check_model(model):
    
    if model is None:
        raise ValueError("Model is not trained")

    
def check_node(model, node_id):

    check_model(model)

    total_nodes=model.tree_.node_count

    if node_id < 0 or node_id >= total_nodes:
        raise ValueError(f"invalid node id: {node_id}")

def check_leaf_node(model, node_id):    
    if model.tree_.children_left[node_id]== -1 and model.tree_.children_left[node_id]== model.tree_.children_right[node_id]:
         raise ValueError("Cannot operate on a leaf node")
        

def check_feature_exists(X, feature):
    if feature not in X.columns:
        raise ValueError(f"feature:{feature} is not available")


def check_variable_exists(X, var_list):
    for var in var_list:
        if var not in X.columns:
            raise ValueError(f"variable:{var} is not available")

def check_suggestions(suggestions):
    if suggestions is None:
        raise ValueError(f"Suggestions cannot be None. Generate suggestions using suggest_split() first")

    if not isinstance(suggestions, pd.DataFrame):
        raise TypeError("Suggestions must be a pandas DataFrame")

    if suggestions.empty:
        raise ValueError(f"Suggestions cannot be an empty Dataframe")




            

