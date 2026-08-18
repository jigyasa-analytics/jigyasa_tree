from sklearn.tree import _tree
import matplotlib.pyplot as plt
import numpy as np

def get_node_counts(tree_structure, node_id):
   value=tree_structure.value[node_id].flatten()
   samples= tree_structure.n_node_samples[node_id]
   if np.isclose(value.sum(),1.0):
      value=value * samples

   return value

def plot_tree(model,feature_names,impurity=True, proportion=True, title=None, target=None,facecolor="white",
              edgecolor="black", fontcolor="black",gradient=False):

    
    class_names = model.classes_

    tree = model.tree_
    tree_text=[]   
        


    def probs(v):
        return v / np.sum(v)

    def gini(v):
        p = probs(v)
        return 1 - np.sum(p**2)

    def entropy(v):
        p = probs(v)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    def logworth(v):
        p = probs(v)
        return -np.sum(p * np.log(p + 1e-9))

    def class_block(v):
        p = probs(v)
        lines = []
        for i, cname in enumerate(class_names):
            count=v[i]
            bar = "█" * int(p[i] * 15)

            if proportion==False:
                lines.append(f"{cname:<5} {int(count):<4} {bar:<15}")
            else:
                lines.append(f"{cname:<5} {int(count):<4} ({p[i]:<4.2f}) {bar:<15}")
        return "\n".join(lines)


    

    pos = {}
    x_counter = 0

    def assign_pos(node, depth=0):
        nonlocal x_counter

        if node == -1:
            return

        if tree.feature[node] == _tree.TREE_UNDEFINED:
            pos[node] = (x_counter, depth)
            x_counter += 1
            return

        assign_pos(tree.children_left[node], depth + 1)

        pos[node] = (x_counter, depth)
        x_counter += 1

        assign_pos(tree.children_right[node], depth + 1)


    def draw(node, parent=None, is_left=None):

        CMAPS={"blue":plt.cm.Blues,
               "green": plt.cm.Greens,
               "grey": plt.cm.Greys,
               "orange":plt.cm.Oranges,
               "purple": plt.cm.Purples,
               "yellow":plt.cm.YlOrBr_r,
               "red": plt.cm.Reds,
               "lightgreen":plt.cm.RdYlGn}
               
        if node == -1:
            return

        x, y = pos[node]
        v = get_node_counts(tree, node)
        n = tree.n_node_samples[node]

        impurity_text=""
        if impurity:
            impurity_text=f"""
Gini     : {gini(v):.3f}
Entropy  : {entropy(v):.3f}
LogWorth : {logworth(v):.3f}
"""
            


        if node == 0:
            text = f"""ALL ROWS
    ────────────────────
    Node ID : {node:<5}
    Level   : {y:<5}
    Samples : {n:<5}
    Target  : {target:<9}

    CLASS PROB
{class_block(v)}
    {impurity_text}"""
           
        else:
            
            text = f"""Node ID : {node:<9}
Level   : {y:<5}
Samples : {n:<5}"""
                        
            if parent is not None and parent != -1:
                f = feature_names[tree.feature[parent]]
                t = tree.threshold[parent]

                if is_left:
                    rule = f"{f} ≤ {t:.2f}"
                else:
                    rule = f"{f} > {t:.2f}"

                text = f"""{rule}
    ────────────────────
    {text}

    CLASS PROB
{class_block(v)}
    {impurity_text}"""

                        
            if tree.feature[node] == _tree.TREE_UNDEFINED:
                if parent is not None and parent != -1:
                    f = feature_names[tree.feature[parent]]
                    t = tree.threshold[parent]

                    if is_left:
                        rule = f"{f} ≤ {t:.2f}"
                    else:
                        rule = f"{f} > {t:.2f}"
                else:
                    rule = "ROOT"

                text = f"""LEAF ({rule})
    ────────────────────
    Node ID : {node:<5}
    Level   : {y:<5}
    Samples : {n:<5}

    CLASS PROB
{class_block(v)}
    {impurity_text}"""

        
         
        if gradient:
            cmap=CMAPS.get(facecolor.lower(),plt.cm.RdYlGn)
            fc=cmap(np.max(probs(v)))
        else:
            fc=facecolor

        x_scale=2.5
        y_scale=3.5  
        
        txt=ax.text(
            x * x_scale, -y * y_scale,
            text,
            ha="center",
            va="center",
            multialignment="center",
            fontsize=8,
            family="monospace",
            color = fontcolor, 
            bbox=dict(
                boxstyle="round",
                fc = fc,  
                ec = edgecolor
            )
        )

        tree_text.append(txt)


        left = tree.children_left[node]
        right = tree.children_right[node]
        

        if left != -1:
            x2, y2 = pos[left]
            ax.plot([x * x_scale, x2 * x_scale], [-y * y_scale, -y2 * y_scale], 'k-', lw=1.2)
            draw(left, node, True)

        if right != -1:
            x2, y2 = pos[right]
            ax.plot([x * x_scale, x2 * x_scale], [-y * y_scale, -y2 * y_scale], 'k-', lw=1.2)
            draw(right, node, False)


    assign_pos(0)

    xs=[x for x, _ in pos.values()]
    ys=[y for _, y in pos.values()]

    xmin, xmax = min(xs), max(xs)
    ymax = max(ys)

    tree_width= xmax - xmin + 1
    tree_height= ymax + 1

    fig_width= max(18, tree_width * 2.5)
    fig_height= max(8, tree_height * 3)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")    

    ax.set_xlim(xmin * 3 - 1, xmax * 3 + 1)
    ax.set_ylim(-(ymax * 4 + 1) , 1)

    if len(pos) == 1:
        ax.set_xlim(-5,5)
        ax.set_ylim(-4,2)
        
    fig.subplots_adjust(top=0.85)    
    
    draw(0)

    return{
        'figure': fig,
           'axis': ax,
           'positions': pos,
           'tree': tree
           }         
           
