from sklearn.tree import _tree

import matplotlib.pyplot as plt
import numpy as np
from .._fontfit import fit_fontsize


def plot_tree(
    model,
    feature_names,
    impurity=True,
    proportion=True,
    title=None,
    target=None,
    facecolor="white",
    edgecolor="black",
    fontcolor="black",
    gradient=False,
    ax=None,
    fontsize=None,
    node_data=None,
    sample_weight=None,
    node_criteria=None,
):
       
    tree = model.tree_


    if ax is None:
        ax = plt.gca()

    fig = ax.figure

    ax.clear()
    ax.axis("off")

    if title is not None:
        fig.suptitle(title)

    # Recalculate every displayed metric from the rows reaching each node.
    # Do not use tree.impurity: it may represent a different graft criterion.
    target_values = None if node_data is None else node_data[target].to_numpy(dtype=float)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if target_values is not None and weights is not None and len(weights) != len(target_values):
        raise ValueError("sample_weight and node_data must have the same number of rows")

    def node_metrics(node):
        if target_values is None:
            return None
        membership = node_data[f"NO{node}"].to_numpy() == 1
        y = target_values[membership]
        w = np.ones(len(y), dtype=float) if weights is None else weights[membership]
        positive = w > 0
        y, w = y[positive], w[positive]
        if len(y) == 0:
            return None

        mean = np.average(y, weights=w)
        mse = float(np.average((y - mean) ** 2, weights=w))
        order = np.argsort(y, kind="stable")
        sorted_y, sorted_w = y[order], w[order]
        # Any weighted median minimizes absolute error; for a half-weight tie,
        # use the midpoint, as sklearn's median prediction does.
        cumulative = np.cumsum(sorted_w)
        midpoint = 0.5 * sorted_w.sum()
        idx = int(np.searchsorted(cumulative, midpoint, side="left"))
        median = float(sorted_y[idx])
        if np.isclose(cumulative[idx], midpoint) and idx + 1 < len(sorted_y):
            median = float((sorted_y[idx] + sorted_y[idx + 1]) / 2)
        mae = float(np.average(np.abs(y - median), weights=w))

        # sklearn's poisson node impurity is half the mean Poisson deviance.
        # It is undefined for negative targets or a non-positive node mean.
        poisson = None
        if np.all(y >= 0) and mean > 0:
            positive_y = y > 0
            terms = np.full(len(y), mean, dtype=float) - y
            terms[positive_y] += y[positive_y] * np.log(y[positive_y] / mean)
            poisson = float(np.average(terms, weights=w))
        return mse, float(np.sqrt(mse)), mae, poisson

    def regression_block(v):
        return f":{v   :.2f}" 


    subtree_width = {}

    def calculate_subtree_width(node):

        if node == -1:
            return 0

        left = tree.children_left[node]
        right = tree.children_right[node]

        # Leaf node
        if (
            left == _tree.TREE_LEAF
            and right == _tree.TREE_LEAF
        ):

            subtree_width[node] = 1.0

            return 1.0

        left_width = calculate_subtree_width(
            left
        )

        right_width = calculate_subtree_width(
            right
        )

        width = (
            left_width
            + right_width
        )

        subtree_width[node] = width

        return width

    calculate_subtree_width(0)

    pos = {}

    def assign_positions(
        node,
        left_boundary,
        right_boundary,
        depth=0,
    ):

        if node == -1:
            return

        left = tree.children_left[node]
        right = tree.children_right[node]


        if (
            left == _tree.TREE_LEAF
            and right == _tree.TREE_LEAF
        ):

            x = (
                left_boundary
                + right_boundary
            ) / 2.0

            pos[node] = (
                x,
                depth,
            )

            return


        left_width = subtree_width[left]
        right_width = subtree_width[right]

        total_child_width = (
            left_width
            + right_width
        )

        split = (
            left_boundary
            + (
                right_boundary
                - left_boundary
            )
            * left_width
            / total_child_width
        )

        assign_positions(
            left,
            left_boundary,
            split,
            depth + 1,
        )

        assign_positions(
            right,
            split,
            right_boundary,
            depth + 1,
        )

        x_left = pos[left][0]
        x_right = pos[right][0]

        pos[node] = (
            (x_left + x_right) / 2.0,
            depth,
        )

    total_width = subtree_width[0]

    assign_positions(
        0,
        0.0,
        total_width,
        0,
    )


    max_depth = max(
        depth
        for _, depth in pos.values()
    )

    x_margin = max(
        total_width * 0.02,
        0.25,
    )

    ax.set_xlim(
        -x_margin,
        total_width + x_margin,
    )

    ax.set_ylim(
        max_depth + 0.75,
        -0.75,
    )

    CMAPS = {
        "blue": plt.cm.Blues,
        "green": plt.cm.Greens,
        "grey": plt.cm.Greys,
        "orange": plt.cm.Oranges,
        "purple": plt.cm.Purples,
        "yellow": plt.cm.YlOrBr_r,
        "red": plt.cm.Reds,
        "lightgreen": plt.cm.RdYlGn,
    }

    def build_node_text(
        node,
        parent=None,
        is_left=None,
    ):

        _, level = pos[node]

        v = tree.value[node].item()
        # Keep the existing node target label; all impurity metrics below are
        # independently computed for every reachable original or grafted node.
        criterion = (node_criteria or {}).get(node, model.get_params().get("criterion", "squared_error"))
        target_label = "Median Target Value" if criterion == "absolute_error" else "Mean Target Value"
        n = tree.n_node_samples[node]

        impurity_text = ""
        if impurity:
            metrics = node_metrics(node)
            if metrics is None:
                impurity_text = "\nMSE : N/A\nRMSE : N/A\nMAE : N/A\nPoisson Impurity : N/A"
            else:
                mse, rmse, mae, poisson = metrics
                impurity_text = (f"\nMSE : {mse:.2f}"
                                 f"\nRMSE : {rmse:.2f}"
                                 f"\nMAE : {mae:.2f}")
                if poisson is not None:
                    impurity_text += f"\nPoisson Impurity : {poisson:.2f}"
                else:
                    impurity_text += "\nPoisson Impurity : N/A"

        if node == 0:

            target_text = f"Target: {target}" if target else ""
            root_text = f"""Node ID : {node:<9}
Level   : {level:<5}
Samples : {n:<5}"""
            text = f"""ALL ROWS
    ────────────────────
    {root_text}
   {target_text}    
    {target_label} {regression_block(v)}
    {impurity_text} """            
            
            return text


        if (
            parent is not None
            and parent != -1
        ):

            feature = (
                feature_names[
                    tree.feature[parent]
                ]
            )

            threshold = (
                tree.threshold[parent]
            )

            if is_left:

                rule = (
                    f"{feature} "
                    f"≤ {threshold:.2f}"
                )

            else:

                rule = (
                    f"{feature} "
                    f"> {threshold:.2f}"
                )

        else:

            rule = "ROOT"

        node_text=f"""Node ID : {node:<9}
Level   : {level:<5}
Samples : {n:<5}"""
        if (
            tree.feature[node] == _tree.TREE_UNDEFINED ):            
            text = f"""LEAF({rule})
    ────────────────────
    {node_text}
    {target_label} {regression_block(v)}
    {impurity_text}"""

            return text

        text = f"""{rule}
    ────────────────────
    {node_text}
    {target_label} {regression_block(v)}
    {impurity_text}"""

        return text

    def draw_edges(node):

        left = tree.children_left[node]
        right = tree.children_right[node]

        x1, y1 = pos[node]


        if left != _tree.TREE_LEAF:

            x2, y2 = pos[left]

            ax.plot(
                [x1, x2],
                [y1, y2],
                color=edgecolor,
                linewidth=0.5,
                zorder=1,
            )

            draw_edges(left)


        if right != _tree.TREE_LEAF:

            x2, y2 = pos[right]

            ax.plot(
                [x1, x2],
                [y1, y2],
                color=edgecolor,
                linewidth=0.5,
                zorder=1,
            )

            draw_edges(right)

    draw_edges(0)


    tree_text = []

    # Gradient colours scale each node's mean target between the tree's
    # smallest and largest node values.
    node_values = tree.value[: tree.node_count].reshape(-1)
    value_min = float(np.min(node_values))
    value_range = float(np.max(node_values)) - value_min

    def draw_nodes(
        node,
        parent=None,
        is_left=None,
    ):

        x, y = pos[node]

        v = tree.value[node].item()

        text = build_node_text(
            node,
            parent,
            is_left,
        )

        if gradient:

            cmap = CMAPS.get(
                str(facecolor).lower(),
                plt.cm.RdYlGn,
            )

            fc = cmap(
                (v - value_min) / value_range
                if value_range > 0
                else 0.5
            )

        else:

            fc = facecolor

 
        initial_fontsize = (
            fontsize
            if fontsize is not None
            else 8.0
        )

        txt = ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            multialignment="center",
            fontsize=initial_fontsize,
            family="monospace",
            color=fontcolor,
            bbox=dict(
                boxstyle="round",
                fc=fc,
                ec=edgecolor,
                linewidth=0.5,
            ),
            zorder=2,
        )

        tree_text.append(txt)

        left = tree.children_left[node]
        right = tree.children_right[node]

        if left != _tree.TREE_LEAF:

            draw_nodes(
                left,
                node,
                True,
            )



        if right != _tree.TREE_LEAF:

            draw_nodes(
                right,
                node,
                False,
            )

    draw_nodes(0)

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.02,
        top=(
            0.92
            if title
            else 0.98
        ),
    )

    
    fitted_fontsize = fontsize

    if fontsize is None and len(tree_text) > 0:
        fitted_fontsize = fit_fontsize(fig, ax, tree_text)


    return {
        "figure": fig,
        "axis": ax,
        "positions": pos,
        "tree": tree,
        "texts": tree_text,
        "fontsize": fitted_fontsize,
    }
