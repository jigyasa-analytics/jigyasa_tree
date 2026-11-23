# jigyasa_tree

jigyasa_tree is a Python library for building, visualizing, pruning and interactively refining decision trees for classification and regression.
It provides tools for inspecting tree structure, generating rules, manually adjusting splits, creating custom subtrees and improving model explainability.

---

## Features

 - Train Decision tree models using the scikit-learn backend
 - Support classification and regression
 - Visualize Decision Trees
 - Generate Decision Rules
 - Inspect node-level data
 - Manually prune nodes
 - Suggest candidate splits at a pruned node (one-level search over selected variables with configurable split parameters)
 - Build custom depth-1 subtrees
 - Support optional sample weights during training and split search
 - Save and load trained models
 
---

## Installation

### Install from Source

Clone the repository and install the package locally:

```bash
git clone https://github.com/jigyasa-analytics/jigyasa_tree.git
cd jigyasa_tree
pip install .
```

---

## Full Workflow Example - Classification
```python
from jigyasa_tree import JigyasaTreeClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

data=load_iris()
    
df=pd.DataFrame(data.data, columns=data.feature_names)
df["target"]=data.target

#Train test split
train_data, val_data=train_test_split(
    df,
    test_size=0.2,
    random_state=42,
     stratify=df["target"]
    )


X_train=train_data.drop(columns="target")
y_train=train_data["target"]

X_val=val_data.drop(columns="target")
y_val=val_data["target"]

tree_params={'criterion': 'entropy', 'min_samples_leaf':0.10, 'random_state':42, 'max_depth': 3}
target="target"

#Create the Classifier
tree=JigyasaTreeClassifier(target_name=target,**tree_params)

#Fit the tree
tree.fit(train_data, val_data, X_train, y_train, X_val, y_val)

#Generate plot
plot_result=tree.plot()
fig=plot_result['figure']
plt.show()

#Prune a node
tree.prune(node_id=2)

#select candidate variables for split suggestion
var_list= X_train.columns[0:2].to_list()
print(f"Variable List:{var_list}")

#Suggest alternative split at the pruned node
best_split, suggestions=tree.suggest_splits(var_list=var_list, node_id=2, criterion="entropy")
print(f"Best Split:{best_split['feature']}")

#Create subtree using the recommended feature
tree.create_subtree(node_id=2, var_to_add=best_split["feature"])

#Save and load tree
tree.save("tree_model.pkl")

loaded_tree=JigyasaTreeClassifier.load("tree_model.pkl")

#Generate plot for the loaded (manually edited) tree
plot_result=loaded_tree.plot()
fig=plot_result['figure']
plt.show()
```
---

## Full Workflow Example - Regression
```python
from jigyasa_tree import JigyasaTreeRegressor
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd

data=load_diabetes()
    
df=pd.DataFrame(data.data, columns=data.feature_names)
df["target"]=data.target

#Train test split
train_data, val_data=train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    )


X_train=train_data.drop(columns="target")
y_train=train_data["target"]

X_val=val_data.drop(columns="target")
y_val=val_data["target"]

tree_params={'criterion': 'squared_error', 'min_samples_leaf':0.10, 'random_state':42, 'max_depth': 3}
target="target"

#Create the regressor
tree=JigyasaTreeRegressor(target_name=target,**tree_params)

#Fit the tree
tree.fit(train_data, val_data, X_train, y_train, X_val, y_val)

#Generate plot
plot_result=tree.plot()
fig=plot_result['figure']
plt.show()

#Prune a node
tree.prune(node_id=1)

#select candidate variables for split suggestion
var_list= X_train.columns[0:2].to_list()
print(f"Variable List:{var_list}")

#Suggest alternative split at the pruned node
best_split, suggestions=tree.suggest_splits(var_list=var_list, node_id=1, criterion="absolute_error")
print(f"Best Split:{best_split['feature']}")

#Create subtree using the recommended feature
tree.create_subtree(node_id=1, var_to_add=best_split["feature"])

#Save and load tree
tree.save("tree_model.pkl")

loaded_tree=JigyasaTreeRegressor.load("tree_model.pkl")

#Generate plot for the loaded (manually edited) tree
plot_result=loaded_tree.plot()
fig=plot_result['figure']
plt.show()

```
---
## Import

The classifier and regressor can be imported directly from `jigyasa_tree`:

```python
from jigyasa_tree import JigyasaTreeClassifier, JigyasaTreeRegressor
```
---

## Core API

### JigyasaTreeClassifier

```python
JigyasaTreeClassifier(target_name, **tree_params)
```
Creates a decision tree for classification. `target_name` (the name of the target column in `train_data` and `val_data`) is required; any other keyword arguments are passed to scikit-learn's `DecisionTreeClassifier`.

---

### JigyasaTreeRegressor

```python
JigyasaTreeRegressor(target_name, **tree_params)
```
Creates a decision tree for regression. `target_name` is required; any other keyword arguments are passed to scikit-learn's `DecisionTreeRegressor`. The target can be any numeric type (integer or float).

---
## Methods

The following methods are available for working with the decision trees

### Fit the tree

```python
tree.fit(train_data, val_data, X_train, y_train, X_val, y_val, sample_weight=None)
```

Fits the model, generates decision rules and adds tree-node variables to the training and validation datasets.

`train_data` must contain exactly the same rows as `X_train` plus the target column (and the same for `val_data`, `X_val`, `y_val`); otherwise `fit()` raises a `ValueError`.

---

### Plot the tree

```python
plot_result= tree.plot()
```
Returns a dictionary:
- `plot_result['figure']` -> matplotlib figure object
- `plot_result['axis']` -> matplotlib axes
- `plot_result['fontsize']` -> font size used for the node boxes

Options:

| Parameter | Default | Description |
|---|---|---|
| `impurity` | `True` | Show node metrics (classifier: Gini, entropy, log loss; regressor: MSE, RMSE, MAE, and Poisson impurity) |
| `proportion` | `True` | Show class proportions in each node (classifier) |
| `title` | `None` | Figure title |
| `facecolor` | `"white"` | Node colour; with `gradient=True` one of `"blue"`, `"green"`, `"grey"`, `"orange"`, `"purple"`, `"yellow"`, `"red"`, `"lightgreen"` |
| `edgecolor` | `"black"` | Node border colour |
| `fontcolor` | `"black"` | Text colour |
| `gradient` | `False` | Shade nodes by class purity (classifier) or by mean target value (regressor) |
| `ax` | `None` | Existing matplotlib axes to draw on |
| `fontsize` | `None` | Fixed font size; by default the largest size at which no boxes overlap is chosen |

Example: Save and display the tree

```python
plot_result= tree.plot()
fig = plot_result['figure']
fig.savefig("plot_default.png")
plt.show() 
```
Example: Plot with a custom figure size and title

```python
fig, ax=plt.subplots(figsize=(18,16))
plot_result=tree.plot(ax=ax, title="Title")
fig=plot_result['figure']
fig.savefig("plot_custom.png")
plt.show() 
```
---
### Prune a node of the tree

```python
tree.prune(node_id)
```

Prunes the subtree rooted at the specified node, converting the selected node into a leaf node. `node_id` must be a whole number and a node that is still in the tree.

---

### Suggest alternative splits at a pruned node

```python
best_split, suggestions=tree.suggest_splits(var_list, node_id, **split_params)
```
Returns:
- best_split: best feature recommendation
- suggestions: DataFrame of candidate splits, ranked best first

The node must first be pruned using `prune(node_id)`. The method fits depth-1 candidate trees using the selected variables and ranks valid splits based on weighted impurity reduction. Alternative splits are generated through successive feature removal.

Users can specify scikit-learn decision-tree parameters through `**split_params`, except `max_depth`, which is fixed at 1.

Example (Classifier):

```python
best_split, suggestions = tree.suggest_splits(
    var_list=var_list, node_id=2, criterion="gini", min_samples_leaf=0.05
)
tree.create_subtree(node_id=2, var_to_add=best_split["feature"])
```

Example (Regressor):

```python
best_split, suggestions = tree.suggest_splits(
    var_list=var_list, node_id=1, criterion="absolute_error", min_samples_leaf=0.05
)
tree.create_subtree(node_id=1, var_to_add=best_split["feature"])
```

---

### Create a subtree at a pruned node

```python
tree.create_subtree(node_id, var_to_add)
```

Attaches a depth-1 subtree using the selected feature (`var_to_add`) at the pruned node (`node_id`). The feature must be present in the suggestions generated for that node, and reusable child nodes must be available.

---

### Get tree rules

```python
rule_strings=tree.get_rules()
```
Returns:
- rule_strings: List of human readable decision rules extracted from the tree

---

### Get train and validation data with tree node variables

```python
train_data, val_data=tree.get_node_data()
```
Returns:
- train_data: train data with node variables
- val_data: validation data with node variables

---

### Get tree rules and add tree node variables to new dataset

```python
node_data, rules = tree.get_tree_rules_and_add_node_variables(data)
```
Returns:
- node_data: data with node variables
- rules: List of human readable decision rules extracted from the tree

`data` must contain every feature used by the tree, with numeric values and no missing values.

---

### Get the model and its parameters

```python
model=tree.get_model()     #the underlying scikit-learn tree (after any manual changes)
params=tree.get_params()   #the tree parameters passed when the tree was created
```

---

### Save and Load a Tree Model

Saves the current tree model to the specified file path. Missing directories are created automatically.

```python
# Save and load a classifier
classifier_tree.save("saved_models/classifier_tree.pkl")
loaded_classifier = JigyasaTreeClassifier.load("saved_models/classifier_tree.pkl")

# Save and load a regressor
regressor_tree.save("saved_models/regressor_tree.pkl")
loaded_regressor = JigyasaTreeRegressor.load("saved_models/regressor_tree.pkl")
```
---

## Logging

The library is silent by default. To see progress messages (for example, which node was pruned), turn on logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

---

## Running the tests

```bash
pip install ".[dev]"
pytest
```
---
## Requirements

This package requires the following dependencies:

| Dependency | Required Version |
|---|---|
| Python | `>=3.10,<3.15` |
| NumPy | `>=1.23,<1.25` for Python < 3.12; `>=2.0` for Python >= 3.12 and < 3.15 |
| Pandas | `>=2.2,<2.4` |
| Scikit-learn | `>=1.2,<1.8` |
| Matplotlib | `>=3.6,<4.0` |
| Joblib | `>=1.2,<1.6` |
| Packaging | `>=21.0` |

---


## Limitations and Design Constraints

### 1. Subtree Restrictions
 - Cannot attach a subtree to an original LEAF node
 - Only a pruned node can be replaced
 - Suggestions must be generated for the same node before creating a subtree
 - Manual replacement subtrees have a fixed depth of 1 and require reusable original child-node IDs

### 2. Tree Growth Constraint
 - After pruning, tree can regrow only up to the depth of the original tree

### 3. Data Requirements
 - No missing or infinite values allowed
 - All features must be numeric or encoded
 - Column names of the form `NO<number>` (for example `NO1`) are reserved for the node variables

### 4. Target Support
 - Only single-target classification and regression are currently supported.
 - Some split parameters require newer scikit-learn versions; consult the installed version before using `monotonic_cst`.

---

## License

This project is licensed under MIT License.

See LICENSE file for details.

