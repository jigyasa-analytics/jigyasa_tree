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
git clone
https://github.com/jigyasa-analytics/jigyasa_tree.git
cd jigyasa_tree
pip install .
```

---

## Full Workflow Example
```python

from jigyasa_tree import Jigyasa_Tree
from matplotlib.pyplot as plt

#Create Jigyasa_Tree
tree=Jigyasa_Tree(target_name=target,criterion = 'entropy', min_samples_leaf = 0.10, random_state = 42)

#Fit the tree
tree.fit(train_data, val_data, X_train,y_train, X_val, y_val)

#Generate plot
plot_result=tree.plot()
fig=plot_result['figure']
plt.show()

#Prune a node
tree.prune(node_id=0)

#Suggest splits with other features at pruned node
var_list= X_train.columns[2:4].to_list()
best_split, suggestions=tree.suggest_splits(var_list=var_list, node_id=0)

#Create subtree and add to node_id
tree.create_subtree(node_id=0, var_to_add=best_split["feature"])

#Save and load tree
tree.save("tree_model.pkl")

loaded_tree=Jigyasa_Tree.load("tree_model.pkl")
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

 - python >= 3.10
 - numpy >= 1.23
 - pandas >= 2.2
 - scikit-learn >= 1.2
 - matplotlib >= 3.6
 - joblib >= 1.2

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

