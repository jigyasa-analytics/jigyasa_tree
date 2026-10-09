from sklearn.tree import _tree

import matplotlib.pyplot as plt
import numpy as np


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
       
    tree = model.tree_


    if ax is None:
        ax = plt.gca()

    fig = ax.figure

    ax.clear()
    ax.axis("off")

    if title is not None:
        fig.suptitle(title)

    def mse(v):
        v1=np.mean((v-np.mean(v))**2)
        return v1   


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

        n = tree.n_node_samples[node]


        impurity_text = ""

        if impurity:

            impurity_text = (
                f"\n"
                f"MSE : {tree.impurity[node]:.2f}\n"
                f"RMSE : {np.sqrt(tree.impurity[node]):.2f}"
            )

        if node == 0:

            target_text = f"Target: {target:<15}" if target else ""
            root_text = f"""Node ID : {node:<9}
Level   : {level:<5}
Samples : {n:<5}"""
            text = f"""ALL ROWS
    ────────────────────
    {root_text}
    {target_text}    
    Mean Target Value {regression_block(v)}
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
    Mean Target Value {regression_block(v)}
    {impurity_text}"""

            return text

        text = f"""{rule}
    ────────────────────
    {node_text}
    Mean Target Value {regression_block(v)}
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
