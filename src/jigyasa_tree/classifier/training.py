
from sklearn.tree import DecisionTreeClassifier

def train_tree(X_train, y_train, sample_weight=None, **tree_params):

    tree_model = DecisionTreeClassifier(**tree_params)
    tree_model.fit(X_train, y_train, sample_weight=sample_weight)

    return tree_model


            

