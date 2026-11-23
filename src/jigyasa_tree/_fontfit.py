import numpy as np


def fit_fontsize(fig, ax, texts, start=8.0, minimum=1.0,
                 max_width_ratio=0.35, max_height_ratio=0.40):
    """Find the largest font size (<= start) at which no node box overlaps
    another and none is too large for the axes.

    Text boxes grow in proportion to the font size, so the boxes are measured
    once at ``start`` and the scale that removes every overlap is computed
    directly, instead of redrawing the whole figure for every candidate size.
    """
    if not texts:
        return start

    for txt in texts:
        txt.set_fontsize(start)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    boxes = [txt.get_bbox_patch().get_window_extent(renderer) for txt in texts]
    cx = np.array([(b.x0 + b.x1) / 2 for b in boxes])
    cy = np.array([(b.y0 + b.y1) / 2 for b in boxes])
    w = np.array([b.width for b in boxes])
    h = np.array([b.height for b in boxes])

    axes_box = ax.get_window_extent(renderer)
    scale = 1.0

    # Boxes must not be larger than a share of the axes.
    if w.max() > 0:
        scale = min(scale, axes_box.width * max_width_ratio / w.max())
    if h.max() > 0:
        scale = min(scale, axes_box.height * max_height_ratio / h.max())

    # Two boxes stop overlapping once they are separated on either axis.
    n = len(boxes)
    if n > 1:
        i, j = np.triu_indices(n, k=1)
        half_w = (w[i] + w[j]) / 2
        half_h = (h[i] + h[j]) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            sx = np.where(half_w > 0, np.abs(cx[i] - cx[j]) / half_w, np.inf)
            sy = np.where(half_h > 0, np.abs(cy[i] - cy[j]) / half_h, np.inf)
        pair_scale = np.maximum(sx, sy)
        if pair_scale.size:
            scale = min(scale, float(pair_scale.min()))

    # Small safety margin so boxes do not touch exactly.
    size = max(minimum, min(start, start * scale * 0.97))

    for txt in texts:
        txt.set_fontsize(size)
    fig.canvas.draw()
    return size
