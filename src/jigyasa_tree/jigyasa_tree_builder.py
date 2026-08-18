from .training import train_tree
from .features import add_node_variables
from .plotting import plot_tree
from .pruning import prune_node
from .split_search import suggest_splits
from .subtree import create_manual_subtree
from .persistence import save_tree, load_tree
from .validation import check_training_data,check_val_data,check_model,check_node,check_feature_exists,check_leaf_node,check_variable_exists,check_suggestions
import copy

class Jigyasa_Tree:

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


    def fit(self, train_data, val_data, X_train, y_train, X_val, y_val):

        self.train_data=train_data
        self.val_data= val_data

        check_training_data(X_train,y_train)
        check_val_data(X_val,y_val)
        
        self.X_train=X_train
        self.y_train=y_train

        self.X_val=X_val
        self.y_val=y_val

        self.feature_names=list(X_train.columns)
        self.target_names=list(y_train.unique())

        self.model=train_tree(X_train,y_train, **self.tree_params)
        self.org_model=copy.deepcopy(self.model)

        self.node_train, self.rules_train=add_node_variables(self.model,train_data, self.feature_names)

        self.node_val, self.rules_val=add_node_variables(self.model,val_data, self.feature_names)

        return self


    def plot(self, impurity=True, proportion=True, facecolor="white", edgecolor="black", fontcolor="black",gradient=False):

        check_model(self.model)

        self.plot_result=plot_tree(model=self.model, feature_names=self.feature_names, impurity=impurity, proportion=proportion,
                                   target=self.target_name, facecolor=facecolor, edgecolor=edgecolor, fontcolor=fontcolor,gradient=gradient)


        return self.plot_result
    

    def prune(self, node_id):

        check_node(self.model, node_id)
        check_leaf_node(self.model, node_id)
        
        self.model=prune_node(self.model, node_id)

        self.node_train, self.rules_train=add_node_variables(self.model,self.train_data, self.feature_names)

        self.node_val, self.rules_val=add_node_variables(self.model,self. val_data, self.feature_names)        

        return self
    

    def suggest_splits(self,var_list, node_id):

        check_node(self.model, node_id)
        check_variable_exists(self.X_train,var_list)

        self.var_list=var_list        

        best_split,self.suggestions=suggest_splits(self.node_train, self.target_name, self.var_list, node_id)

        return best_split, self.suggestions

    def create_subtree(self, node_id, var_to_add):

        check_node(self.model, node_id)
        check_leaf_node(self.org_model, node_id)
        check_feature_exists(self.X_train,var_to_add)
        check_suggestions(self.suggestions)

        self.model= create_manual_subtree(self.model,node_id,var_to_add, self.suggestions, self.org_model)

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
        node_data, rules = add_node_variables(self.model, data, self.feature_names)
        return node_data, rules
    
    def save(self, path):
        save_tree(self, path)

    @staticmethod
    def load(path):
        return load_tree(path)

           

