from sklearn.tree import _tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn import __version__ as sklearn_version
from packaging.version import Version

def use_proportions():
    return Version(sklearn_version) >= Version("1.4")

def get_node_counts(tree_structure, node_id):    
    value = tree_structure.value[node_id].flatten().copy()
    samples = tree_structure.n_node_samples[node_id]
    if use_proportions():
        value = value * samples

    return value


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
):


    class_names = model.classes_
    tree = model.tree_

    if ax is None:
        ax = plt.gca()

    fig = ax.figure

    ax.clear()
    ax.axis("off")

    if title is not None:
        fig.suptitle(title)


    def probs(v):
        total = np.sum(v)

        if total == 0:
            return np.zeros_like(v, dtype=float)

        return v / total

    def gini(v):
        p = probs(v)

        return 1 - np.sum(p ** 2)

    def entropy(v):
        p = probs(v)
        p = p[p > 0]
        if len(p) == 0:
            return 0.0
        return -np.sum(p * np.log2(p))

    def logloss(v):
        p = probs(v)
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    def class_block(v):
        p = probs(v)

        lines = []

        name_width=max(8, max(len(str(c)) for c in class_names))
        for i, cname in enumerate(class_names):

            count = v[i]

            bar = "█" * int(
                p[i] * 15
            )

            if proportion is False:
                lines.append(
                    f"  {str(cname):<{name_width}}"
                    f"  {int(round(count)):>8}"
                    f"  {bar:<15}"
                )

            else:
                lines.append(
                    f"  {str(cname):<{name_width}}"
                    f"  {int(round(count)):>8}"
                    f"  ({p[i]:>4.4f})"
                    f"  {bar:<15}"
                )

        return "\n".join(lines)


    subtree_width = {}

    def calculate_subtree_width(node):

        if node == -1:
            return 0

        left = tree.children_left[node]
        right = tree.children_right[node]


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

        v = get_node_counts(
            tree,
            node,
        )

        n = tree.n_node_samples[node]


        impurity_text = ""

        if impurity:

            impurity_text = (
                f"\n"
                f"{'Gini': <10} : {gini(v):.3f}\n"
                f"{'Entropy': <10} : {entropy(v):.3f}\n"
                f"{'LogLoss': <10} : {logloss(v):.3f}"
            )


        if node == 0:

            text = (
                f"""ALL ROWS
    ────────────────────
    Node ID : {node:<5}
    Level   : {level:<5}
    Samples : {n:<5}
    Target  : {target:<9}

    CLASS PROB
{class_block(v)}
    {impurity_text}"""
            )

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



        if (
            tree.feature[node]
            == _tree.TREE_UNDEFINED
        ):

            text = (
                f"""LEAF ({rule})
    ────────────────────
    Node ID : {node:<5}
    Level   : {level:<5}
    Samples : {n:<5}

    CLASS PROB
{class_block(v)}
    {impurity_text}"""
            )

            return text


        text = (
            f"""{rule}
    ────────────────────
    Node ID : {node:<5}
    Level   : {level:<5}
    Samples : {n:<5}

    CLASS PROB
{class_block(v)}
    {impurity_text}"""
        )

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

    def draw_nodes(
        node,
        parent=None,
        is_left=None,
    ):

        x, y = pos[node]

        v = get_node_counts(
            tree,
            node,
        )

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
                np.max(probs(v))
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

    if (fontsize is None and len(tree_text) > 0):

        current_fontsize = 8.0
        min_fontsize = 1.0
        reduction_factor = 0.90

        max_width_ratio = 0.35
        max_height_ratio = 0.40

        while (current_fontsize >= min_fontsize ):
            
            for txt in tree_text:
                txt.set_fontsize(current_fontsize)
                
            fig.canvas.draw()

            renderer = fig.canvas.get_renderer()            

            boxes = []

            for txt in tree_text:

                bbox = (
                    txt
                    .get_bbox_patch()
                    .get_window_extent(
                        renderer
                    )
                )

                boxes.append(bbox)

            axes_bbox= ax.get_window_extent(renderer)
            available_width= axes_bbox.width
            available_height= axes_bbox.height

            oversized= any(box.width > available_width * max_width_ratio
                           or
                           box.height > available_height * max_height_ratio
                           for box in boxes)                           


            overlap_found = False

            number_of_boxes = len(boxes)

            for i in range(number_of_boxes):

                for j in range(
                    i + 1,
                    number_of_boxes
                ):

                    if boxes[i].overlaps( boxes[j]):

                        overlap_found = True
                        break

                if overlap_found:
                    break


            if not overlap_found and not oversized:
                break


            current_fontsize *= (
                reduction_factor
            )


        current_fontsize = max(
            current_fontsize,
            min_fontsize,
        )


        for txt in tree_text:

            txt.set_fontsize(
                current_fontsize
            )

        fitted_fontsize = (
            current_fontsize
        )

        fig.canvas.draw()    


    return {
        "figure": fig,
        "axis": ax,
        "positions": pos,
        "tree": tree,
        "texts": tree_text,
        "fontsize": fitted_fontsize,
    }
