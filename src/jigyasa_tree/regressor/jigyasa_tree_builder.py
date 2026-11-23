from .training import train_tree
from .features import add_node_variables
from .plotting import plot_tree
from .pruning import prune_node
from .split_search import suggest_splits
from .subtree import create_manual_subtree
from .persistence import save_tree, load_tree
from .validation import (
    check_training_data,
    check_val_data,
    check_target_name,
    check_data_matches,
    check_no_node_columns,
    check_new_data,
    check_model,
    check_node,
    check_feature_exists,
    check_leaf_node,
    check_pruned_node,
    check_variable_exists,
    check_suggestions,
)
import copy
import numpy as np


class JigyasaTreeRegressor:

    def __init__(
        self,
        target_name=None,
        **tree_params
        ):

        #sklearn Tree parameters
        self.tree_params=tree_params

        #model objects
        self.org_model=None

        self.model=None

        self.rules_train=None

        self.node_train=None

        self.rules_val=None

        self.node_val=None

        self.train_data=None

        self.val_data=None

        self.X_train=None

        self.y_train=None

        self.feature_names=None

        self.target_name=target_name

        self.X_val=None

        self.y_val=None

        self.plot_result=None

        self.var_list=None

        self.suggestions=None

        #node the stored suggestions were generated for
        self.suggestions_node=None
        self.sample_weight=None


    def fit(self, train_data, val_data, X_train, y_train, X_val, y_val, sample_weight=None):

        check_target_name(self.target_name, train_data, val_data)
        check_training_data(X_train,y_train)
        check_val_data(X_val,y_val)
        check_data_matches(train_data, X_train, y_train, self.target_name, "train")
        check_data_matches(val_data, X_val, y_val, self.target_name, "val")
        check_no_node_columns(train_data)
        check_no_node_columns(val_data)

        self.train_data=train_data
        self.val_data= val_data

        self.X_train=X_train
        self.y_train=y_train

        self.X_val=X_val
        self.y_val=y_val

        self.feature_names=list(X_train.columns)
        self.target_names=list(y_train.unique())

        # Keep weights aligned by row position, including duplicate DataFrame indices.
        if sample_weight is None:
            self.sample_weight=None
        else:
            weights=np.asarray(sample_weight, dtype=float)
            if weights.ndim != 1 or len(weights) != len(X_train):
                raise ValueError("sample_weight must be a 1D array with one weight per training row")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError("sample_weight must be finite, nonnegative, and have positive total")
            self.sample_weight=weights.copy()

        self.model=train_tree(X_train,y_train, sample_weight=self.sample_weight, **self.tree_params)
        self.org_model=copy.deepcopy(self.model)

        self.suggestions=None
        self.suggestions_node=None

        self.node_train, self.rules_train=add_node_variables(self.model,train_data, self.feature_names)

        self.node_val, self.rules_val=add_node_variables(self.model,val_data, self.feature_names)

        return self


    def plot(self, impurity=True, proportion=True, title=None, facecolor="white", edgecolor="black", fontcolor="black",gradient=False, ax=None, fontsize=None):

        check_model(self.model)

        self.plot_result=plot_tree(model=self.model, feature_names=self.feature_names, impurity=impurity, proportion=proportion, title=title,
                                   target=self.target_name, facecolor=facecolor, edgecolor=edgecolor, fontcolor=fontcolor,gradient=gradient, ax=ax, fontsize=fontsize, node_data=self.node_train,
                                   sample_weight=self.sample_weight)


        return self.plot_result


    def prune(self, node_id):

        check_node(self.model, node_id)
        check_leaf_node(self.model, node_id)

        self.model=prune_node(self.model, node_id)

        #the tree changed, so earlier suggestions no longer apply
        self.suggestions=None
        self.suggestions_node=None

        self.node_train, self.rules_train=add_node_variables(self.model,self.train_data, self.feature_names)

        self.node_val, self.rules_val=add_node_variables(self.model,self.val_data, self.feature_names)

        return self


    def suggest_splits(self,var_list, node_id, **split_params):

        check_node(self.model, node_id)
        check_pruned_node(self.model, self.org_model, node_id)
        check_variable_exists(self.X_train,var_list)

        # A user-selected criterion is allowed; default to the original criterion.
        from .validation import check_split_parameters
        check_split_parameters(split_params, kind="regressor", feature_count=len(var_list))
        split_params = {"criterion": self.model.criterion, **split_params}

        self.var_list=var_list

        best_split,self.suggestions=suggest_splits(self.node_train, self.target_name, self.var_list, node_id, sample_weight=self.sample_weight, **split_params)
        self.suggestions_node=node_id

        return best_split, self.suggestions

    def create_subtree(self, node_id, var_to_add):

        check_node(self.model, node_id)
        check_pruned_node(self.model, self.org_model, node_id)
        check_feature_exists(self.X_train,var_to_add)
        check_suggestions(self.suggestions, getattr(self, "suggestions_node", None), node_id, var_to_add)

        self.model = create_manual_subtree(self.model, node_id, var_to_add, self.suggestions, self.org_model)

        #suggestions have been used; the tree changed
        self.suggestions=None
        self.suggestions_node=None

        self.node_train, self.rules_train=add_node_variables(self.model,self.train_data, self.feature_names)

        self.node_val, self.rules_val=add_node_variables(self.model,self.val_data, self.feature_names)

        return self


    def get_rules(self):
        return self.rules_train


    def get_node_data(self):
        return self.node_train, self.node_val


    def get_model(self):
        return self.model


    def get_params(self):
        return self.tree_params


    def get_tree_rules_and_add_node_variables(self, data):
        check_model(self.model)
        check_new_data(data, self.feature_names)
        node_data, rules = add_node_variables(self.model, data, self.feature_names)
        return node_data, rules

    def save(self, path):
        save_tree(self, path)

    @classmethod
    def load(cls, path):
        tree = load_tree(path)
        if not isinstance(tree, cls):
            raise TypeError(
                f"{path} contains a {type(tree).__name__}, not a {cls.__name__}"
            )
        return tree
