"""Visualize pruning policies with a Hinton-like diagram.

"""
import numpy as np
import pylab as pl
from numpy import array, sqrt, zeros_like
from nltk import Tree
from ldp.parsing.util import item_tree
from ldp.math.util import sigmoid
from arsenal.viz.util import update_ax


#from ldp.rl.policy import det_mask
def det_mask(example, theta):
    "Return mask under deterministic policy."
    features = example.features
    mask = example.mask
    for x in example.nodes:
        mask[x] = (features[x].dot(theta) >= 0)
    return mask


def get_spans(t):
    got = set()
    if isinstance(t, Tree):
        for s in item_tree(t).subtrees():
            (_,I,K) = s.label()
            got.add((I,K))
    return got


# access to each example's axes
def show_charts(examples, theta, parse):
    axes = show_charts.axes
    for e in examples:
        if e.sentence not in axes:
            axes[e.sentence] = pl.figure().add_subplot(111)
        show_mask(e, parse=parse, theta=theta, ax=axes[e.sentence])
show_charts.axes = {}


# given mask
def show_mask(e, mask=None, parse=None, theta=None, title=None, ax=None):

    if ax is None:
        ax = pl.figure().add_subplot(111)

    if title is None:
        title = '%s' % e.sentence

    if mask is None:
        assert theta is not None
        mask = det_mask(e, theta)

    got_spans = []
    if parse is not None:
        _, _, t = parse.rollout2(e, mask)
        got_spans = get_spans(t)

    if theta is None:
        q = zeros_like(e.mask, dtype=float)
        for x in e.nodes:
            q[x] = mask[x] - 0.5
    else:
        q = zeros_like(e.mask, dtype=float)
        for x in e.nodes:
            q[x] = sigmoid(e.features[x].dot(theta)) - 0.5

    hinton(ax, q, e.gold_spans, got_spans, maxWeight=None, title=title)


def hinton(ax, W, gold, got, maxWeight=None, title=''):
    """Visualize pruning mask and parser output. Produces a image which is similar
    to an (upper-triangular) Hinton diagram.

    Size and color indicate pruning strength and whether hard decisions is
    keep/prune (white/black, respectively).

    Additional annotations

     * gold box -- unlabled span in "gold" standard tree

     * Parser output

        - green border: unlabeled item is correct.

        - red border: unlabeled is not correct.

    TODO:

      - How do I indicate errors in current parse? current visualization only
        shows which unlabeled spans are good? In many cases unlabeled recall is
        perfect, but labeled recall is quite low, e.g., 60%.

    """

    N = W.shape[0]

    if ax is None:
        ax = pl.figure().add_subplot(111)

    def blob(x, y, area, c):
        """
        Draws a square-shaped blob with the given area (< 1) at
        the given coordinates.
        """
        hs = sqrt(area) / 2
        xcorners = array([x - hs, x + hs, x + hs, x - hs])
        ycorners = array([y - hs, y - hs, y + hs, y + hs])
        ax.fill(xcorners, ycorners, c, edgecolor=c)

    with update_ax(ax):
        height, width = W.shape
        if maxWeight is None:
            maxWeight = np.max(np.abs(W)) / .5

        ax.fill(array([1, width, width, 1]),
                array([0, 0, height, height]), 'gray')
        ax.axis('off')
        ax.axis('equal')

        for y in xrange(height):
            for x in xrange(y, width):
                _x = x+1
                _y = y+1
                w = W[y,x]

                if abs(x-y) > 1 and abs(x-y) != N:
                    if (y, x) in got:
                        if (y,x) in gold:
                            c = 'green'
                        else:
                            c = 'red'
                        blob(_x - 0.5, height - _y + 0.5, 0.85, c)

                if (y, x) in gold or abs(x-y) == 1 or abs(x-y) == N:
                    blob(_x - 0.5, height - _y + 0.5, 0.70, 'yellow')

                if abs(x - y) in (0, 1, height):
                    continue

                if w >= 0:
                    blob(_x - 0.5, height - _y + 0.5, min(1, w / maxWeight), 'white')
                else:
                    blob(_x - 0.5, height - _y + 0.5, min(1, -w / maxWeight), 'black')

        if title:
            ax.set_title(title)
