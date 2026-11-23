"""Tests for jigyasa_tree. Run from the repository folder with:  pytest"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_iris
from sklearn.model_selection import train_test_split

from jigyasa_tree import JigyasaTreeClassifier, JigyasaTreeRegressor


# ---------- helpers ----------

def frame(loader):
    d = loader()
    df = pd.DataFrame(d.data, columns=d.feature_names)
    df["target"] = d.target
    return df


def split(df, stratify=True):
    train, val = train_test_split(
        df, test_size=0.25, random_state=0,
        stratify=df["target"] if stratify else None)
    return (train, val, train.drop(columns="target"), train["target"],
            val.drop(columns="target"), val["target"])


def fit(cls, df, stratify=True, **params):
    tree = cls(target_name="target", **{"max_depth": 3, "random_state": 0, **params})
    tree.fit(*split(df, stratify))
    return tree


def internal_nodes(tree):
    t = tree.model.tree_
    return [n for n in range(1, t.node_count) if t.children_left[n] != -1]


def leaves(tree):
    t = tree.model.tree_
    reach, stack = set(), [0]
    while stack:
        n = stack.pop()
        if n == -1:
            continue
        reach.add(n)
        stack += [t.children_left[n], t.children_right[n]]
    return [n for n in reach if t.children_left[n] == -1]


def assert_tree_consistent(tree, data):
    """Every row is in exactly one leaf, and it is the leaf scikit-learn uses."""
    node_data, _ = tree.get_tree_rules_and_add_node_variables(data)
    cols = [f"NO{n}" for n in leaves(tree)]
    assert (node_data[cols].sum(axis=1) == 1).all()
    flagged = node_data[cols].idxmax(axis=1).str[2:].astype(int).to_numpy()
    assert (tree.model.apply(data[tree.feature_names]) == flagged).all()


@pytest.fixture(scope="module")
def iris():
    return frame(load_iris)


@pytest.fixture(scope="module")
def cancer():
    return frame(load_breast_cancer)


@pytest.fixture(scope="module")
def diabetes():
    return frame(load_diabetes)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------- README workflows ----------

@pytest.mark.parametrize("cls,loader,node,stratify", [
    (JigyasaTreeClassifier, load_iris, 2, True),
    (JigyasaTreeRegressor, load_diabetes, 1, False),
])
def test_readme_workflow(tmp_path, cls, loader, node, stratify):
    df = frame(loader)
    tree = fit(cls, df, stratify, min_samples_leaf=0.10,
               criterion="entropy" if cls is JigyasaTreeClassifier else "squared_error")
    assert tree.plot()["figure"] is not None
    tree.prune(node_id=node)
    best, suggestions = tree.suggest_splits(var_list=tree.feature_names[:2], node_id=node)
    assert not suggestions.empty
    tree.create_subtree(node_id=node, var_to_add=best["feature"])
    assert_tree_consistent(tree, tree.train_data)

    path = tmp_path / "tree_model.pkl"
    tree.save(path)
    loaded = cls.load(path)
    assert np.allclose(loaded.model.predict(tree.X_val), tree.model.predict(tree.X_val))
    loaded.plot()


def test_no_console_output(capsys, iris):
    tree = fit(JigyasaTreeClassifier, iris)
    tree.prune(2)
    best, _ = tree.suggest_splits(tree.feature_names, 2)
    tree.create_subtree(2, best["feature"])
    assert capsys.readouterr().out == ""


def test_rule_strings_reproduce_node_flags(cancer):
    tree = fit(JigyasaTreeClassifier, cancer, max_depth=4)
    df = tree.train_data.copy()
    for rule in tree.get_rules():
        exec(rule, {}, {"df": df})
    node_train, _ = tree.get_node_data()
    for col in [c for c in node_train.columns if c.startswith("NO")]:
        assert (df[col].fillna(0) == node_train[col]).all()


# ---------- bugs B1-B10 from the review ----------

def test_b1_regressor_accepts_integer_target(diabetes):
    df = diabetes.copy()
    df["target"] = df["target"].astype(int)
    fit(JigyasaTreeRegressor, df, stratify=False)


def test_b2_regressor_gradient_plot(diabetes):
    tree = fit(JigyasaTreeRegressor, diabetes, stratify=False)
    tree.plot(gradient=True, facecolor="blue")


def test_b3_suggestions_for_other_node_refused(cancer):
    tree = fit(JigyasaTreeClassifier, cancer, max_depth=4)
    first = internal_nodes(tree)[0]
    t = tree.model.tree_
    other = [n for n in internal_nodes(tree)
             if n not in (t.children_left[first], t.children_right[first])][-1]
    tree.prune(first)
    tree.prune(other)
    tree.suggest_splits(tree.feature_names[:5], first)
    with pytest.raises(ValueError, match="made for node"):
        tree.create_subtree(other, tree.feature_names[0])


def test_b4_unpruned_node_refused(cancer):
    tree = fit(JigyasaTreeClassifier, cancer, max_depth=4)
    node = internal_nodes(tree)[0]
    with pytest.raises(ValueError, match="has not been pruned"):
        tree.suggest_splits(tree.feature_names[:5], node)
    with pytest.raises(ValueError, match="has not been pruned"):
        tree.create_subtree(node, tree.feature_names[0])


def test_b5_removed_node_refused(cancer):
    tree = fit(JigyasaTreeClassifier, cancer, max_depth=4)
    t = tree.model.tree_
    node = next(n for n in internal_nodes(tree) if t.children_left[t.children_left[n]] != -1)
    child = int(t.children_left[node])
    tree.prune(node)
    with pytest.raises(ValueError, match="no longer part of the tree"):
        tree.prune(child)


def test_b6_new_data_with_missing_values_refused(iris):
    tree = fit(JigyasaTreeClassifier, iris)
    new = iris.copy()
    new.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        tree.get_tree_rules_and_add_node_variables(new)


def test_b6_new_data_missing_column_refused(iris):
    tree = fit(JigyasaTreeClassifier, iris)
    with pytest.raises(ValueError, match="missing feature columns"):
        tree.get_tree_rules_and_add_node_variables(iris.drop(columns=[tree.feature_names[2]]))


def test_b7_reserved_column_name_refused(iris):
    df = iris.copy()
    df["NO1"] = 999.0
    with pytest.raises(ValueError, match="reserved"):
        fit(JigyasaTreeClassifier, df)


def test_b8_rules_valid_with_quotes_in_names(iris):
    df = iris.copy()
    df.columns = ["it's", 'pet"al', "a b", "x[0]", "target"]
    df["it's"] = df["a b"]
    tree = fit(JigyasaTreeClassifier, df)
    for rule in tree.get_rules():
        compile(rule, "rule", "exec")


def test_b9_target_name_required(iris):
    tree = JigyasaTreeClassifier(max_depth=3)
    with pytest.raises(ValueError, match="target_name (is required|must be a nonempty string)"):
        tree.fit(*split(iris))


def test_b10_train_data_must_match(iris):
    train, val, X_train, y_train, X_val, y_val = split(iris)
    tree = JigyasaTreeClassifier(target_name="target", max_depth=3)
    with pytest.raises(ValueError, match="same rows"):
        tree.fit(val, val, X_train, y_train, X_val, y_val)


# ---------- other improvements ----------

def test_validation_set_with_one_class_allowed(cancer):
    train, val, X_train, y_train, X_val, y_val = split(cancer)
    keep = y_val == 1
    tree = JigyasaTreeClassifier(target_name="target", max_depth=3)
    tree.fit(train, val[keep], X_train, y_train, X_val[keep], y_val[keep])


@pytest.mark.parametrize("bad", ["text", "inf"])
def test_bad_feature_values_clear_error(iris, bad):
    df = iris.copy()
    if bad == "text":
        df["colour"] = np.where(df.index % 2 == 0, "red", "blue")
        match = "numeric"
    else:
        df.iloc[3, 0] = np.inf
        match = "infinite"
    with pytest.raises(ValueError, match=match):
        fit(JigyasaTreeClassifier, df)


@pytest.mark.parametrize("node_id", ["2", 2.5, None, True, -1, 999])
def test_invalid_node_ids_clear_error(iris, node_id):
    tree = fit(JigyasaTreeClassifier, iris)
    with pytest.raises(ValueError):
        tree.prune(node_id)


def test_feature_not_in_suggestions_clear_error(cancer):
    tree = fit(JigyasaTreeClassifier, cancer)
    node = internal_nodes(tree)[0]
    tree.prune(node)
    tree.suggest_splits(tree.feature_names[:2], node)
    with pytest.raises(ValueError, match="not among the suggested splits"):
        tree.create_subtree(node, tree.feature_names[10])


def test_no_possible_split_clear_error(iris):
    tree = fit(JigyasaTreeClassifier, iris.assign(const=1.0))
    tree.prune(2)
    with pytest.raises(ValueError, match="No valid split"):
        tree.suggest_splits(["const"], 2)


def test_load_wrong_class_refused(tmp_path, diabetes):
    tree = fit(JigyasaTreeRegressor, diabetes, stratify=False)
    tree.save(tmp_path / "reg.pkl")
    with pytest.raises(TypeError):
        JigyasaTreeClassifier.load(tmp_path / "reg.pkl")


def test_save_creates_nested_folders(tmp_path, iris):
    tree = fit(JigyasaTreeClassifier, iris)
    path = tmp_path / "a" / "b" / "c" / "model.pkl"
    tree.save(path)
    assert path.exists()


def test_root_prune_and_rebuild(iris):
    tree = fit(JigyasaTreeClassifier, iris)
    tree.prune(0)
    best, _ = tree.suggest_splits(tree.feature_names, 0)
    tree.create_subtree(0, best["feature"])
    assert_tree_consistent(tree, tree.train_data)


def test_plot_speed_large_tree():
    tree = fit(JigyasaTreeClassifier, frame(load_digits), max_depth=5)
    start = time.time()
    tree.plot()
    assert time.time() - start < 60
    
    
@pytest.mark.parametrize("cls,loader,stratify", [
    (JigyasaTreeClassifier, load_iris, True),
    (JigyasaTreeRegressor, load_diabetes, False),
])
def test_backward_compatibility(cls, loader, stratify):

    tree = fit(cls, frame(loader), stratify=stratify)

    node = internal_nodes(tree)[0]
    tree.prune(node)

    best, _ = tree.suggest_splits(tree.feature_names[:2], node)

    # Simulate an old model without suggestions_node
    del tree.suggestions_node

    with pytest.raises(ValueError, match="node associated with the stored suggestions is unknown"):
        tree.create_subtree(node, best["feature"])
        
        
def test_classifier_regression_target_warning(diabetes):

    df = diabetes.copy()

    # Diabetes targets are whole-number float values
    assert df["target"].nunique() > 20

    with pytest.warns(UserWarning, match="looks like a regression target"):
        fit(JigyasaTreeClassifier, df, stratify=False)


# ---------- configurable depth-1 split search and independent subtree criteria ----------

@pytest.mark.parametrize("cls,loader,stratify,original,subtree", [
    (JigyasaTreeClassifier, load_iris, True, "gini", "entropy"),
    (JigyasaTreeClassifier, load_iris, True, "entropy", "log_loss"),
    (JigyasaTreeClassifier, load_iris, True, "log_loss", "gini"),
    (JigyasaTreeRegressor, load_diabetes, False, "squared_error", "absolute_error"),
    (JigyasaTreeRegressor, load_diabetes, False, "absolute_error", "friedman_mse"),
    (JigyasaTreeRegressor, load_diabetes, False, "friedman_mse", "poisson"),
])
def test_subtree_criterion_can_differ_from_original(cls, loader, stratify, original, subtree):
    """Explicit subtree criterion is independent of the original fitted criterion."""
    tree = fit(cls, frame(loader), stratify=stratify, criterion=original)
    tree.prune(0)
    features = tree.feature_names[:4]
    best, suggestions = tree.suggest_splits(features, 0, criterion=subtree)
    assert best["feature"] in suggestions["Feature Name"].values
    assert suggestions["Feature Name"].is_unique
    assert suggestions["Impurity_Reduction"].is_monotonic_decreasing
    assert len(suggestions) <= len(features)
    assert tree.model.criterion == original
    tree.create_subtree(0, best["feature"])
    assert_tree_consistent(tree, tree.train_data)
    assert np.isfinite(tree.model.predict(tree.X_val)).all()


@pytest.mark.parametrize("cls,loader,stratify,original", [
    (JigyasaTreeClassifier, load_iris, True, "entropy"),
    (JigyasaTreeRegressor, load_diabetes, False, "absolute_error"),
])
def test_subtree_criterion_defaults_to_original(cls, loader, stratify, original):
    tree = fit(cls, frame(loader), stratify=stratify, criterion=original)
    tree.prune(0)
    best, suggestions = tree.suggest_splits(tree.feature_names[:4], 0)
    assert best["feature"] in suggestions["Feature Name"].values
    assert not suggestions.empty


@pytest.mark.parametrize("cls,loader,stratify", [
    (JigyasaTreeClassifier, load_iris, True),
    (JigyasaTreeRegressor, load_diabetes, False),
])
def test_split_search_rejects_depth_override(cls, loader, stratify):
    """Manual subtree creation currently supports depth-one splits only."""
    tree = fit(cls, frame(loader), stratify=stratify)
    tree.prune(0)
    with pytest.raises(ValueError, match="max_depth"):
        tree.suggest_splits(tree.feature_names[:3], 0, max_depth=2)


@pytest.mark.parametrize("cls,loader,stratify", [
    (JigyasaTreeClassifier, load_iris, True),
    (JigyasaTreeRegressor, load_diabetes, False),
])
@pytest.mark.parametrize("kwargs,exception", [
    ({"min_samples_leaf": -1}, ValueError),
    ({"min_samples_split": 1}, ValueError),
    ({"splitter": "invalid"}, ValueError),
    ({"max_leaf_nodes": 1}, ValueError),
    ({"monotonic_cst": [1]}, ValueError),
    ({"not_a_tree_parameter": 5}, TypeError),
])
def test_split_search_parameter_validation(cls, loader, stratify, kwargs, exception):
    tree = fit(cls, frame(loader), stratify=stratify)
    tree.prune(0)
    with pytest.raises(exception):
        tree.suggest_splits(tree.feature_names[:3], 0, **kwargs)


@pytest.mark.parametrize("cls,loader,stratify,criterion", [
    (JigyasaTreeClassifier, load_iris, True, "gini"),
    (JigyasaTreeRegressor, load_diabetes, False, "absolute_error"),
])
def test_configured_split_can_create_consistent_subtree(cls, loader, stratify, criterion):
    tree = fit(cls, frame(loader), stratify=stratify, criterion=criterion)
    tree.prune(0)
    best, suggestions = tree.suggest_splits(
        tree.feature_names[:4], 0, min_samples_leaf=2, random_state=0
    )
    assert not suggestions.empty
    tree.create_subtree(0, best["feature"])
    assert tree.model.tree_.children_left[0] != -1
    assert tree.model.criterion == criterion
    assert_tree_consistent(tree, tree.train_data)


@pytest.mark.parametrize("cls,loader,stratify", [
    (JigyasaTreeClassifier, load_iris, True),
    (JigyasaTreeRegressor, load_diabetes, False),
])
def test_fit_sample_weight_and_split_search(cls, loader, stratify):
    """Weights are accepted at fit time and do not break manual split suggestions."""
    df = frame(loader)
    parts = split(df, stratify=stratify)
    weights = np.ones(len(parts[0]), dtype=float)
    weights[:max(1, len(weights) // 5)] = 2.0
    tree = cls(target_name="target", max_depth=3, random_state=0)
    tree.fit(*parts, sample_weight=weights)
    tree.prune(0)
    best, suggestions = tree.suggest_splits(tree.feature_names[:4], 0)
    assert best["feature"] in suggestions["Feature Name"].values


@pytest.mark.parametrize("cls,loader,stratify", [
    (JigyasaTreeClassifier, load_iris, True),
    (JigyasaTreeRegressor, load_diabetes, False),
])
def test_fit_invalid_sample_weight_rejected(cls, loader, stratify):
    parts = split(frame(loader), stratify=stratify)
    tree = cls(target_name="target", max_depth=3, random_state=0)
    with pytest.raises(ValueError, match="sample_weight"):
        tree.fit(*parts, sample_weight=np.ones(len(parts[0]) - 1))


# ---------- extended integration and regression coverage ----------

@pytest.mark.parametrize("cls,loader,stratify,criteria", [
    (JigyasaTreeClassifier, load_iris, True, ["gini", "entropy", "log_loss"]),
    (JigyasaTreeRegressor, load_diabetes, False, ["squared_error", "friedman_mse", "absolute_error", "poisson"]),
])
def test_all_supported_criteria(cls, loader, stratify, criteria):
    """Each supported criterion works when selected for the original tree."""
    for criterion in criteria:
        tree = fit(cls, frame(loader), stratify=stratify, criterion=criterion)
        tree.prune(0)
        best, suggestions = tree.suggest_splits(tree.feature_names[:4], 0)
        assert best["feature"] in suggestions["Feature Name"].tolist()
        assert tree.model.criterion == criterion
        tree.create_subtree(0, best["feature"])
        assert tree.model.criterion == criterion
        assert_tree_consistent(tree, tree.train_data)


@pytest.mark.parametrize("cls,loader,stratify", [
    (JigyasaTreeClassifier, load_iris, True),
    (JigyasaTreeRegressor, load_diabetes, False),
])
def test_sequential_feature_removal_search(monkeypatch, cls, loader, stratify):
    """Each round fits ALL remaining features and removes the chosen root feature."""
    import importlib
    module = importlib.import_module("jigyasa_tree.classifier.split_search" if cls is JigyasaTreeClassifier
                                     else "jigyasa_tree.regressor.split_search")
    estimator_name = "DecisionTreeClassifier" if cls is JigyasaTreeClassifier else "DecisionTreeRegressor"
    original = getattr(module, estimator_name)
    seen = []

    class SpyTree(original):
        def fit(self, X, y, *args, **kwargs):
            result = super().fit(X, y, *args, **kwargs)
            chosen = X.columns[int(self.tree_.feature[0])] if self.tree_.feature[0] >= 0 else X.columns[0]
            seen.append((list(X.columns), chosen, self.max_depth))
            return result

    monkeypatch.setattr(module, estimator_name, SpyTree)
    tree = fit(cls, frame(loader), stratify=stratify)
    tree.prune(0)
    variables = tree.feature_names[:4]
    tree.suggest_splits(variables, 0)
    assert len(seen) == len(variables)
    remaining = list(variables)
    for columns, selected, depth in seen:
        assert columns == remaining
        assert depth == 1
        remaining.remove(selected)


@pytest.mark.parametrize("cls,loader,stratify", [
    (JigyasaTreeClassifier, load_iris, True),
    (JigyasaTreeRegressor, load_diabetes, False),
])
def test_save_load_after_manual_edit_preserves_predictions_and_nodes(tmp_path, cls, loader, stratify):
    tree = fit(cls, frame(loader), stratify=stratify)
    tree.prune(0)
    best, _ = tree.suggest_splits(tree.feature_names[:4], 0)
    tree.create_subtree(0, best["feature"])
    before_pred = tree.model.predict(tree.X_val)
    before_nodes = tree.model.apply(tree.X_val)
    path = tmp_path / (cls.__name__ + ".pkl")
    tree.save(path)
    restored = cls.load(path)
    np.testing.assert_array_equal(restored.model.predict(tree.X_val), before_pred)
    np.testing.assert_array_equal(restored.model.apply(tree.X_val), before_nodes)
    assert_tree_consistent(restored, restored.val_data)


@pytest.mark.parametrize("cls,loader,stratify", [
    (JigyasaTreeClassifier, load_iris, True),
    (JigyasaTreeRegressor, load_diabetes, False),
])
def test_suggestions_invalidated_after_edit(cls, loader, stratify):
    tree = fit(cls, frame(loader), stratify=stratify)
    tree.prune(0)
    best, _ = tree.suggest_splits(tree.feature_names[:4], 0)
    tree.create_subtree(0, best["feature"])
    assert tree.suggestions is None
    with pytest.raises(ValueError):
        tree.create_subtree(0, best["feature"])


def test_balanced_classifier_effective_weights_and_subtree():
    """Class balancing is inherited from the ORIGINAL training distribution."""
    from sklearn.utils.class_weight import compute_sample_weight
    df = frame(load_breast_cancer)
    # Deterministic imbalanced subset, retaining both classes.
    df = pd.concat([df[df.target == 0].iloc[:25], df[df.target == 1].iloc[:225]])
    parts = split(df, stratify=True)
    weighted = JigyasaTreeClassifier(target_name="target", max_depth=3,
                                    random_state=0, class_weight="balanced")
    plain = JigyasaTreeClassifier(target_name="target", max_depth=3,
                                 random_state=0)
    weighted.fit(*parts)
    plain.fit(*parts)
    expected = compute_sample_weight("balanced", parts[3])
    np.testing.assert_allclose(weighted.effective_sample_weights, expected)
    np.testing.assert_allclose(plain.effective_sample_weights, np.ones(len(parts[3])))
    weighted.prune(0)
    best, suggestions = weighted.suggest_splits(weighted.feature_names[:5], 0)
    assert not suggestions.empty
    weighted.create_subtree(0, best["feature"])
    assert_tree_consistent(weighted, weighted.train_data)


@pytest.mark.parametrize("cls,loader,stratify", [
    (JigyasaTreeClassifier, load_iris, True),
    (JigyasaTreeRegressor, load_diabetes, False),
])
def test_repeated_prune_suggest_rebuild_root(cls, loader, stratify):
    tree = fit(cls, frame(loader), stratify=stratify)
    for _ in range(2):
        tree.prune(0)
        best, _ = tree.suggest_splits(tree.feature_names[:4], 0)
        tree.create_subtree(0, best["feature"])
        assert_tree_consistent(tree, tree.train_data)


@pytest.mark.parametrize("cls,loader,stratify", [
    (JigyasaTreeClassifier, load_iris, True),
    (JigyasaTreeRegressor, load_diabetes, False),
])
def test_split_search_does_not_mutate_original_training_data(cls, loader, stratify):
    tree = fit(cls, frame(loader), stratify=stratify)
    tree.prune(0)
    before = tree.train_data.copy(deep=True)
    tree.suggest_splits(tree.feature_names[:4], 0)
    pd.testing.assert_frame_equal(tree.train_data, before)


@pytest.mark.parametrize("cls,loader,stratify", [
    (JigyasaTreeClassifier, load_iris, True),
    (JigyasaTreeRegressor, load_diabetes, False),
])
def test_explicit_depth_one_is_not_accepted(cls, loader, stratify):
    """The fixed depth is internal; callers should omit max_depth entirely."""
    tree = fit(cls, frame(loader), stratify=stratify)
    tree.prune(0)
    with pytest.raises(ValueError, match="max_depth"):
        tree.suggest_splits(tree.feature_names[:3], 0, max_depth=1)


def test_regressor_no_possible_split_clear_error(diabetes):
    df = diabetes.copy()
    df["const"] = 1.0

    tree = fit(JigyasaTreeRegressor, df, stratify=False)
    tree.prune(0)

    with pytest.raises(ValueError, match="No valid split found"):
        tree.suggest_splits(["const"], 0)
