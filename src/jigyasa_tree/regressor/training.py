
from sklearn.tree import DecisionTreeRegressor

def train_tree(X_train, y_train, **tree_params):

    tree_model = DecisionTreeRegressor(**tree_params)
    tree_model.fit(X_train, y_train)

    return tree_model


            

