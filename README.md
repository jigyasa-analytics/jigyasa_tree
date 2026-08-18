</>Markdown

## jigyasa_tree

jigyasa_tree is a Python library for building, visualizing, pruning and interactively refining decision tree classifiers. 
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
https://github.com/<username>/<repository>.git
cd <repository>
pip install .
```

---

## Full Workflow Example
```python

from jigyasa_tree import Jigyasa_Tree
from matplotlib.pyplot as plt

#Create Jigyasa_Tree
tree=Jigyasa_Tree(target_name=target,criterion = 'entropy', min_samples_leaf = 0.10, random_state = 42)

#fit
tree.fit(train_data, val_data, X_train,y_train, X_val, y_val)

#generate plot
plot_result=tree.plot()
fig=plot_result['figure']
plt.show()

#PRUNE
tree.prune(node_id=0)

#SUGGEST
var_list= X_train.columns[2:4].to_list()
best_split, suggestions=tree.suggest_splits(var_list=var_list, node_id=0)

#CREATE SUBTREE and add To node_id
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

## Fit 

```python
tree.fit(train_data, val_data, X_train, y_train, X_val, y_val)
```

Fits the model, generate rules and adds tree-node variables to the training and validation datasets

---

## plot

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
## Prune

```python
tree.prune(node_id)
```

Prunes a specified node from the tree

---

## suggest

```python
best_split, suggestions=tree.suggest_splits(var_list, node_id)
```
Returns:
- best_split: best feature recommendation
- suggestions: ranked list of candidate splits

---

## create Manual tree

```python
tree.create_subtree(node_id, feature)
```

Attatch the depth 1 subtree of selected feature to the node_id

---

## Get Tree rules

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

## Running Tests

```bash
pytest
```

---

## Requirements

 -python >= 3.10
 -numpy >= 1.23
 -pandas >= 2.2
 -scikit-learn >= 1.2
 -matplotlib >= 3.6
 -joblib >= 1.2

---

## Project status

 - version 0.1.0
 - status: initial release
 
---

## Limitation and Design Constrains

## 1. Binary Classification Only
 - Supports binary classification only
 - No multi-class support yet
 - No regression support

## 2. Subtree Restriction
 - Cannot attach a subtree to an original LEAF nodes
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

