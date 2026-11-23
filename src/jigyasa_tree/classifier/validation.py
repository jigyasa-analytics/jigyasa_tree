import numbers
import re

import numpy as np
import pandas as pd

import warnings


NODE_COLUMN = re.compile(r"^NO\d+$")


def _check_features(X, label):

    non_numeric = [
        col for col in X.columns
        if not (pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col]))
    ]
    if non_numeric:
        raise ValueError(
            f"{label} features must be numeric or encoded. Non-numeric columns: {non_numeric}"
        )

    if X.isna().any().any():
        raise ValueError(f"{label} features contain missing values")

    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError(f"{label} features contain infinite values")

    if len(X.columns) != len(set(X.columns)):
        raise ValueError("Feature names must be unique")


def check_training_data(X_train, y_train):

    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
        raise TypeError("X_train must be a DataFrame and y_train a Series")
    if X_train.empty:
        raise ValueError("Training data is empty")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of rows")

    _check_features(X_train, "Training")

    if y_train.isna().any():
        raise ValueError("Training target contains missing values")

    if y_train.nunique() < 2:
        raise ValueError("Training target contains only one class")

    if np.issubdtype(y_train.dtype, np.floating):
        if not np.all(np.mod(y_train, 1) == 0):
            raise ValueError("Continuous targets detected. Use JigyasaTreeRegressor for regression")
            
    if np.issubdtype(y_train.dtype, np.floating) and y_train.nunique() > 20:
        warnings.warn(
            f"Target has {y_train.nunique()} distinct values; it looks like a regression target. "
            "Use JigyasaTreeRegressor if you meant regression."
        )


def check_val_data(X_val, y_val):

    if not isinstance(X_val, pd.DataFrame) or not isinstance(y_val, pd.Series):
        raise TypeError("X_val must be a DataFrame and y_val a Series")
    if X_val.empty:
        raise ValueError("Validation data is empty")

    if len(X_val) != len(y_val):
        raise ValueError("X_val and y_val must have the same number of rows")

    _check_features(X_val, "Validation")

    if y_val.isna().any():
        raise ValueError("Validation target contains missing values")

    # A validation set with a single class is allowed (e.g. a small or
    # rare-event validation sample).    

    if np.issubdtype(y_val.dtype, np.floating):
        if not np.all(np.mod(y_val, 1) == 0):
            raise ValueError("Continuous targets detected. Use JigyasaTreeRegressor for regression")
            
    if np.issubdtype(y_val.dtype, np.floating) and y_val.nunique() > 20:
        warnings.warn(
            f"Target has {y_val.nunique()} distinct values; it looks like a regression target. "
            "Use JigyasaTreeRegressor if you meant regression."
        )


def check_target_name(target_name, *datasets):

    if not isinstance(target_name, str) or not target_name.strip():
        raise ValueError("target_name must be a nonempty string")
    if target_name is None:
        raise ValueError(
            "target_name is required, e.g. JigyasaTreeClassifier(target_name='target', ...)"
        )

    for data in datasets:
        if target_name not in data.columns:
            raise ValueError(f"target_name '{target_name}' is not a column of the data passed to fit()")


def check_data_matches(data, X, y, target_name, label):
    """train_data / val_data must hold the same rows as X / y."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{label}_data must be a pandas DataFrame")
    if len(data) != len(X) or not data.index.equals(X.index) or not y.index.equals(X.index):
        raise ValueError(
            f"{label}_data does not have the same rows (length and index) as X_{label}"
        )

    if target_name in X.columns:
        raise ValueError(f"X_{label} must not contain the target column '{target_name}'")

    missing = [col for col in X.columns if col not in data.columns]
    if missing:
        raise ValueError(f"{label}_data is missing feature columns: {missing}")

    if not data[list(X.columns)].equals(X):
        raise ValueError(f"{label}_data feature values do not match X_{label}")

    if not data[target_name].equals(y):
        raise ValueError(f"{label}_data['{target_name}'] does not match y_{label}")


def check_no_node_columns(data):
    """Columns named NO0, NO1, ... are reserved for the node flags."""

    clashes = [col for col in data.columns if NODE_COLUMN.match(str(col))]
    if clashes:
        raise ValueError(
            f"Column names {clashes} are reserved for tree node flags (NO<number>). Please rename them"
        )


def check_new_data(data, feature_names):

    missing = [col for col in feature_names if col not in data.columns]
    if missing:
        raise ValueError(f"Data is missing feature columns used by the tree: {missing}")

    check_no_node_columns(data)
    _check_features(data[feature_names], "Input")


def check_model(model):

    if model is None:
        raise ValueError("Model is not trained. Call fit() first")


def reachable_nodes(model):

    tree = model.tree_
    nodes = set()
    stack = [0]

    while stack:
        node = stack.pop()
        if node == -1 or node in nodes:
            continue
        nodes.add(node)
        stack.append(tree.children_left[node])
        stack.append(tree.children_right[node])

    return nodes


def check_node(model, node_id):

    check_model(model)

    if isinstance(node_id, bool) or not isinstance(node_id, numbers.Integral):
        raise ValueError(f"node_id must be a whole number, got {node_id!r}")

    total_nodes = model.tree_.node_count

    if node_id < 0 or node_id >= total_nodes:
        raise ValueError(f"Invalid node id: {node_id}. Valid ids are 0 to {total_nodes - 1}")

    if node_id not in reachable_nodes(model):
        raise ValueError(
            f"Node {node_id} is no longer part of the tree (it was removed by an earlier prune)"
        )


def check_leaf_node(model, node_id):
    if model.tree_.children_left[node_id] == -1 and model.tree_.children_right[node_id] == -1:
        raise ValueError(f"Cannot operate on leaf node: {node_id}")


def check_pruned_node(model, org_model, node_id):
    """The node must have been split in the original tree and be a leaf now."""

    check_leaf_node(org_model, node_id)

    if model.tree_.children_left[node_id] != -1:
        raise ValueError(
            f"Node {node_id} has not been pruned. Call prune(node_id={node_id}) first"
        )


def check_feature_exists(X, feature):
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' is not available")


def check_variable_exists(X, var_list):
    if isinstance(var_list, (str, bytes)) or not isinstance(var_list, (list, tuple, pd.Index)) or not len(var_list):
        raise ValueError("var_list must be a nonempty sequence of feature names")
    if len(var_list) != len(set(var_list)):
        raise ValueError("var_list contains duplicate features")
    for var in var_list:
        if var not in X.columns:
            raise ValueError(f"Variable '{var}' is not available")


def check_suggestions(suggestions, suggestions_node=None, node_id=None, feature=None):

    if suggestions is None:
        raise ValueError(
            "No suggestions available. Generate them with suggest_splits() for this node first"
        )

    if not isinstance(suggestions, pd.DataFrame):
        raise TypeError("Suggestions must be a pandas DataFrame")

    if suggestions.empty:
        raise ValueError("Suggestions cannot be an empty DataFrame")
        
    if node_id is not None and suggestions_node is None:
        raise ValueError(
            "The node associated with the stored suggestions is unknown. "
            f"Call suggest_splits(var_list, node_id={node_id}) first."
        )

    if node_id is not None and suggestions_node != node_id:
        raise ValueError(
            f"The stored suggestions were made for node {suggestions_node}, not node {node_id}. "
            f"Call suggest_splits(var_list, node_id={node_id}) first"
        )

    if "Feature Name" not in suggestions.columns:
        raise ValueError("Suggestions are missing the 'Feature Name' column")
    if feature is not None and feature not in set(suggestions["Feature Name"]):
        raise ValueError(
            f"Feature '{feature}' is not among the suggested splits: "
            f"{list(suggestions['Feature Name'])}"
        )


def check_split_parameters(split_params, kind, feature_count):
    import inspect
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    estimator = DecisionTreeClassifier if kind == "classifier" else DecisionTreeRegressor
    forbidden = {"max_depth"}
    if kind == "classifier":
        forbidden.add("class_weight")
    allowed = set(inspect.signature(estimator).parameters) - forbidden
    unknown = set(split_params) - allowed - forbidden
    if unknown:
        raise TypeError(f"Unsupported split parameters: {sorted(unknown)}")
    for key in forbidden & set(split_params):
        raise ValueError(f"{key} cannot be overridden for manual subtrees")
    if "monotonic_cst" in split_params:
        mono = split_params["monotonic_cst"]
        if mono is not None:
            if isinstance(mono, (str, bytes)) or np.asarray(mono).ndim != 1 or len(mono) != feature_count:
                raise ValueError("monotonic_cst must be a 1D array matching var_list length")
            if not all(x in (-1, 0, 1) for x in mono):
                raise ValueError("monotonic_cst entries must be -1, 0, or 1")
    for key in ("min_samples_leaf", "min_samples_split"):
        if key in split_params:
            value = split_params[key]
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f"{key} must be a numeric value")
            if isinstance(value, (int, np.integer)):
                minimum = 1 if key == "min_samples_leaf" else 2
                if value < minimum:
                    raise ValueError(f"{key} must be >= {minimum}")
            elif not (0 < value <= 1):
                raise ValueError(f"{key} as a fraction must be in (0, 1]")
    for key in ("min_weight_fraction_leaf", "min_impurity_decrease", "ccp_alpha"):
        if key in split_params and split_params[key] is not None:
            value = split_params[key]
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be a finite nonnegative number")
            if key == "min_weight_fraction_leaf" and value > 0.5:
                raise ValueError("min_weight_fraction_leaf must be <= 0.5")
    if "splitter" in split_params and split_params["splitter"] not in ("best", "random"):
        raise ValueError("splitter must be 'best' or 'random'")
    if "max_leaf_nodes" in split_params and split_params["max_leaf_nodes"] is not None:
        value = split_params["max_leaf_nodes"]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2:
            raise ValueError("max_leaf_nodes must be None or an integer >= 2")
    if "max_features" in split_params and split_params["max_features"] is not None:
        value = split_params["max_features"]
        if isinstance(value, bool) or (isinstance(value, (int, np.integer)) and value < 1) or (isinstance(value, (float, np.floating)) and not (0 < value <= 1)):
            raise ValueError("max_features must be positive and valid for sklearn")
    if "random_state" in split_params and isinstance(split_params["random_state"], bool):
        raise ValueError("random_state cannot be boolean")
