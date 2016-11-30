import pylab as pl
import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull


def point(x, y, angle, length=0.1):
    r = length/np.hypot(angle, 1)
    return (x - r*angle, y + r)


def arrow(x, y, angle, offset, c=None, ax=None):
    ax = ax or pl.gca()
    ax.annotate("", xy=(x,y),
                xytext=point(x, y, angle, length=offset),
                arrowprops=dict(arrowstyle="->", lw=2,
                                color=c,
                                connectionstyle="arc3"))
#                arrowprops=dict(arrowstyle="->, head_width=0.1, head_length=0.1",
#                                lw=2.5, color=c, connectionstyle="arc3"))


def lambda_cone(accuracy, runtime, ax, c, conesize, lines=1, aspect_equal=1):
    #ax.scatter(runtime, accuracy, c=tradeoff, lw=0)

    P = zip(runtime, accuracy)
    #P.extend([(0,0), (runtime.max()+0.5e7, accuracy.max())])
    P.extend([(runtime.min()-.1*runtime.ptp(), 0),
              (runtime.max()+.1*runtime.ptp(), accuracy.max())])

#    tradeoff = np.array(list(tradeoff) + [np.inf, 0.0])

    hull = ConvexHull(P)
    v = hull.points[hull.vertices]

    ddd = []

    V = len(hull.vertices)
    for i in range(V):
        x1,y1 = v[(i-1) % V]
        x,y = v[i]
        x3,y3 = v[(i+1) % V]

        # slopes give a range for lambda values to try.
        m12 = (y-y1)/(x-x1)
        m23 = (y3-y)/(x3-x)

        # Note: This filter skips cases where the convex hull contains a
        # segments (edge) that is beneath the fronter (we know this because of
        # the orientation of the edge)..
        if m12 >= m23:
            continue

        ddd.append([m12, m23, x, y])

        ax.add_artist(Polygon([[x,y],
                               point(x, y, m12, -conesize),
                               point(x, y, m23, -conesize),
                               [x,y]],
                              linewidth=0,
                              alpha=0.5,
                              color=c,
                          ))

        if lines:
            ax.plot([x1,x,x3], [y1,y,y3], c=c, alpha=0.5)

    if aspect_equal:
        ax.set_aspect('equal')
    ax.grid(True)
    return ddd
