## jigyasa_tree

jigyasa_tree is a Python library for building, visualizing, pruning and interactively refining binary decision tree classifiers. 
It provides tools for inspecting tree structure, generating rules, manually adjusting splits and improving model explainability.

---

## Features

 - Train Decision tree models using the scikit-learn backend
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

## Full Workflow Example
```python

from jigyasa_tree import Jigyasa_Tree
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

data=load_iris()
    
df=pd.DataFrame(data.data, columns=data.feature_names)
df["target"]=data.target

#Restrict to binary
df=df[df["target"] != 2]

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

#Create Jigyasa_Tree
tree=Jigyasa_Tree(target_name=target,criterion = 'entropy', min_samples_leaf = 0.10, random_state = 42)

#Fit the tree
tree.fit(train_data, val_data, X_train, y_train, X_val, y_val)

#Generate plot
plot_result=tree.plot()
fig=plot_result['figure']
plt.show()

#Prune a node
tree.prune(node_id=0)

#Variables list to get suggestion
var_list= X_train.columns[0:2].to_list()
print(f"Variable List:{var_list}")

#Suggest splits with other features at pruned node
best_split, suggestions=tree.suggest_splits(var_list=var_list, node_id=0)
print(f"Best Split:{best_split['feature']}")

#Create subtree and add to node_id
tree.create_subtree(node_id=0, var_to_add=best_split["feature"])

#Save and load tree
tree.save("tree_model.pkl")

loaded_tree=Jigyasa_Tree.load("tree_model.pkl")

#Generate plot for the manual tree
plot_result=tree.plot()
fig=plot_result['figure']
plt.show()
```
---

# Core API
## Constructor

```python
Jigyasa_Tree(target_name, **tree_params)
```
---

### Methods

## Fit the tree

```python
tree.fit(train_data, val_data, X_train, y_train, X_val, y_val)
```

Fits the model, generate rules and adds tree-node variables to the training and validation datasets

---

## Plot the tree

```python
plot_result= tree.plot()
```
Returns:
- plot_result['figure'] -> matplotlib figure object

Example

```python
fig = plot_result['figure'] 
```
---
## Prune a node of the tree

```python
tree.prune(node_id)
```

Prunes a specified node from the tree

---

## Suggest splits with other features at pruned node

```python
best_split, suggestions=tree.suggest_splits(var_list, node_id)
```
Returns:
- best_split: best feature recommendation
- suggestions: ranked list of candidate splits

---

## Create manual tree with a new feature at pruned node

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
#save model
tree.save(path)

#load the model
loaded_tree=Jigyasa_Tree.load(path)
```
---
## Requirements

This package requires the following dependencies:

| Dependency | Required Version |
|---|---|
| Python | 3.10 – 3.13 |
| NumPy | `>=1.23,<1.25` for Python < 3.12; `>=2.0` for Python >= 3.12 and < 3.14 |
| Pandas | `>=2.2,<2.4` |
| Scikit-learn | `>=1.2,<1.8` |
| Matplotlib | `>=3.6,<3.7` |
| Joblib | `>=1.2,<1.6` |
---

## Project status

 - version 0.1.0
 - status: initial release
 
---

## Limitation and Design Constraints

## 1. Binary Classification Only
 - Supports binary classification only
 - No multi-class support yet
 - No regression support

## 2. Subtree Restriction
 - Cannot attach a subtree to an original LEAF node
 - Only a pruned node can be replaced

## 3. Tree growth Constraint
 - After pruning, tree can regrow only up to its original structured size

## 4. Data requirements
 - No missing values allowed
 - All features must be numeric or encoded


---

## License

This project is licensed under MIT License.

See LICENSE file for details.

