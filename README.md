## jigyasa_tree

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
 - Suggest optimal splits
 - Build custom subtrees
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
best_split, suggestions=tree.suggest_splits(var_list=var_list, node_id=2)
print(f"Best Split:{best_split['feature']}")

#Create subtree using the recommended feature
tree.create_subtree(node_id=2, var_to_add=best_split["feature"])

#Save and load tree
tree.save("tree_model.pkl")

loaded_tree=JigyasaTreeClassifier.load("tree_model.pkl")

#Generate plot for the manual tree
plot_result=tree.plot()
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
best_split, suggestions=tree.suggest_splits(var_list=var_list, node_id=1)
print(f"Best Split:{best_split['feature']}")

#Create subtree using the recommended feature
tree.create_subtree(node_id=1, var_to_add=best_split["feature"])

#Save and load tree
tree.save("tree_model.pkl")

loaded_tree=JigyasaTreeRegressor.load("tree_model.pkl")

#Generate plot for the manual tree
plot_result=tree.plot()
fig=plot_result['figure']
plt.show()

```
---
# import

The classifier and regressor can be imported directly from 'jigyasa_tree'

```python
from jigyasa_tree import JigyasaTreeClassifier, JigyasaTreeRegressor
```
---

# Core API
## JigyasaTreeClassifier

```python
JigyasaTreeClassifier(target_name, **tree_params)
```
Creates a decision tree for classification
---

## JigyasaTreeRegressor

```python
JigyasaTreeRegressor(target_name, **tree_params)
```
Creates a decision tree for regression
---
# Methods

The following methods are available for working with the decision trees

## Fit the tree

```python
tree.fit(train_data, val_data, X_train, y_train, X_val, y_val)
```

Fits the model, generate decision rules and adds tree-node variables to the training and validation datasets

---

## Plot the tree

```python
plot_result= tree.plot()
```
Returns:
- plot_result['figure'] -> matplotlib figure object

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
## Prune a node of the tree

```python
tree.prune(node_id)
```

Prunes the subtree rooted at the specified node, converting the selected node into a leaf node.

---

## Suggest alternative splits at a pruned node

```python
best_split, suggestions=tree.suggest_splits(var_list, node_id)
```
Returns:
- best_split: best feature recommendation
- suggestions: ranked list of candidate splits

---

## Create a subtree at a pruned node

```python
tree.create_subtree(node_id, feature)
```

Attach the depth 1 subtree of selected feature to the node_id

---

## Get tree rules

```python
rule_strings=tree.get_rules()
```
Returns:
- rule_strings: List of human readable decision rules extracted from the tree

---

## Get train and validation data with tree node variables

```python
train_data, val_data=tree.get_node_data()
```
Returns:
- train_data: train data with node variables
- val_data: validation data with node variables

---

## Get tree rules and add tree node variables to new dataset

```python
node_data, rules = tree.get_tree_rules_and_add_node_variables(data)
```
Returns:
- node_data: data with node variables
- rules: List of human readable decision rules extracted from the tree

---

## Save and Load model

```python
#Saves the current tree model to the specified path
tree.save(path)

#load a classifier
loaded_tree=JigyasaTreeClassifier.load(path)

#Load a regressor
loaded_tree=JigyasaTreeRegressor.load(path)
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
---


## Limitation and Design Constraints

## 1. Subtree Restrictions
 - Cannot attach a subtree to an original LEAF node
 - Only a pruned node can be replaced

## 2. Tree growth Constraint
 - After pruning, tree can regrow only up to the depth of the original tree

## 3. Data requirements
 - No missing values allowed
 - All features must be numeric or encoded

## 4. Target support
 - Only single-target classification and regression are currently supported.

---

## License

This project is licensed under MIT License.

See LICENSE file for details.

