from __future__ import division
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import mesher


def savefig(fname):
    """
    Save a matplotlib figure in 'manuscript/figures'

    Uses the "os" module to specify the path in a cross-platform way.

    Calls plt.savefig in the background.

    Parameters:

    * fname : str
        The file name of the figure **without** the folder path.
        Ex: "my_figure.pdf", not "../manuscript/figures/my_figure.pdf".

    """
    fig_file = os.path.join(os.path.pardir, 'manuscript', 'figures', fname)
    plt.savefig(fig_file, facecolor='w', bbox_inches='tight')


def draw_main_system(ax, length_axes=1, label_size=22, elev=200,
                     azim=-20):
    '''
    Plot the axes forming the main coordinate system.

    Parameters:

    * ax: axes of a matplotlib figure.
    * length_axes: float
        Length of the axes (in meters). Default is 1.
    * label_size: float
        Size of the label font. Default is 22.
    * elev and azim: floats
        Parameters controlling the view of the figure.
        Default is 200 and -20, respectively.
    '''

    # x-axis
    ax.quiver(length_axes, 0, 0, length_axes, 0, 0,
              length=length_axes, color='k', linewidth=2, linestyle='-',
              arrow_length_ratio=0.1)
    ax.text(1.05*length_axes, 0, 0, '$x$', color='k', fontsize=label_size)

    # y-axis
    ax.quiver(0, length_axes, 0, 0, length_axes, 0,
              length=length_axes, color='k', linewidth=2, linestyle='-',
              arrow_length_ratio=0.1)
    ax.text(0, 1.05*length_axes, 0, '$y$', color='k', fontsize=label_size)

    # z-axis
    ax.quiver(0, 0, length_axes, 0, 0, length_axes,
              length=length_axes, color='k', linewidth=2, linestyle='-',
              arrow_length_ratio=0.1)
    ax.text(0, 0, 1.05*length_axes, '$z$', color='k', fontsize=label_size)

    ax.axis('off')

    ax.view_init(elev=elev, azim=azim)


def get_parameters(ellipsoid):
    '''
    Get the coordinate transformation matrix and
    the semi-axes of a given ellipsoid.

    Parameters:

    * ellipsoid : element of :class:`mesher.TriaxialEllipsoid`,
        `mesher.ProlateEllipsoid` or `mesher.OblateEllipsoid`.

    Returns:

    * V: numpy array 2D
        Coordinate transformation matrix from the main coordinate system
        to the local coordinate system. The local coordinate system
        has the origin at the centre of the ellipsoid and the axes
        aligned with the semi-axes of the ellipsoid.
    * a, b, c: floats
        Semi-axes of the ellipsoid.
    * xc, yc, zc: floats
        Coordinates of the elliposid centre referred to the
        main coordinate system.
    '''

    # Coordinate transformation matrix
    V = ellipsoid.transf_matrix

    # Ellipsoid centre
    xc = ellipsoid.x
    yc = ellipsoid.y
    zc = ellipsoid.z

    # Ellipsoid semi-axes
    if ellipsoid.__class__ is mesher.TriaxialEllipsoid:
        a = ellipsoid.large_axis
        b = ellipsoid.intermediate_axis
        c = ellipsoid.small_axis
    if ellipsoid.__class__ is mesher.ProlateEllipsoid:
        a = ellipsoid.large_axis
        b = ellipsoid.small_axis
        c = ellipsoid.small_axis
    if ellipsoid.__class__ is mesher.OblateEllipsoid:
        a = ellipsoid.small_axis
        b = ellipsoid.large_axis
        c = ellipsoid.large_axis

    return V, a, b, c, xc, yc, zc


def draw_ellipsoid(ax, ellipsoid, body_color, body_alpha, npoints=100):
    '''
    Plot the surface of an ellipsoid.

    Parameters:

    * ax: axes of a matplotlib figure.
    * ellipsoid : element of :class:`mesher.TriaxialEllipsoid`,
        `mesher.ProlateEllipsoid` or `mesher.OblateEllipsoid`.
    * body_color: RGB matplotlib tuple
        Color of the body.
    * body_alpha: float
        Transparency of the body.
    * npoints: int
        Number of points used to interpolate the surface
        of the ellipsoid.
    '''

    V, a, b, c, xc, yc, zc = get_parameters(ellipsoid)

    # Spherical angles (in radians) for plotting the ellipsoidal surface.
    u = np.linspace(0, 2 * np.pi, 2*npoints)
    v = np.linspace(0, np.pi, npoints)

    # Cartesian coordinates referred to the body system
    # (https://en.wikipedia.org/wiki/Ellipsoid)
    x1 = a * np.outer(np.cos(u), np.sin(v))
    x2 = b * np.outer(np.sin(u), np.sin(v))
    x3 = c * np.outer(np.ones_like(u), np.cos(v))

    # Cartesian coordinates referred to the main system
    x = V[0, 0]*x1 + V[0, 1]*x2 + V[0, 2]*x3 + xc
    y = V[1, 0]*x1 + V[1, 1]*x2 + V[1, 2]*x3 + yc
    z = V[2, 0]*x1 + V[2, 1]*x2 + V[2, 2]*x3 + zc

    # Plot:
    ax.plot_surface(x, y, z, linewidth=0., color=body_color, alpha=body_alpha)


def draw_axes(ax, ellipsoid, axes_color=(0, 0, 0),
              label_axes=True, label_size=16):
    '''
    Plot three orthogonal axes.

    Parameters:

    * ax: axes of a matplotlib figure.
    * ellipsoid : element of :class:`mesher.TriaxialEllipsoid`,
        `mesher.ProlateEllipsoid` or `mesher.OblateEllipsoid`.
    * label_axes : boolean
        If True, plot the label of all axes.
    * label_size : int
        Define the size of the label of all axes.
    '''

    V, a, b, c, xc, yc, zc = get_parameters(ellipsoid)

    ax.quiver(xc+V[0, 0]*a, yc+V[1, 0]*a, zc+V[2, 0]*a,
              V[0, 0], V[1, 0], V[2, 0],
              length=a, color=axes_color, linewidth=3.0, linestyle='-',
              arrow_length_ratio=0.1)

    ax.quiver(xc+V[0, 1]*b, yc+V[1, 1]*b, zc+V[2, 1]*b,
              V[0, 1], V[1, 1], V[2, 1],
              length=b, color=axes_color, linewidth=3.0, linestyle='-',
              arrow_length_ratio=0.1)

    ax.quiver(xc+V[0, 2]*c, yc+V[1, 2]*c, zc+V[2, 2]*c,
              V[0, 2], V[1, 2], V[2, 2],
              length=c, color=axes_color, linewidth=3.0, linestyle='-',
              arrow_length_ratio=0.1)

    if label_axes is True:

        ax.text(xc+V[0, 0]*a*1.05, yc+V[1, 0]*a*1.05, zc+V[2, 0]*a*1.05,
                '$a \hat{\mathbf{v}}_{1}$', color=axes_color,
                fontsize=label_size)

        ax.text(xc+V[0, 1]*b*1.05, yc+V[1, 1]*b*1.05, zc+V[2, 1]*b*1.05,
                '$b \hat{\mathbf{v}}_{2}$', color=axes_color,
                fontsize=label_size)

        ax.text(xc+V[0, 2]*c*1.05, yc+V[1, 2]*c*1.05, zc+V[2, 2]*c*1.05,
                '$c \hat{\mathbf{v}}_{3}$', color=axes_color,
                fontsize=label_size)


def limits(ax, xmin, xmax, ymin, ymax, zmin, zmax):
    '''
    Set the limits of the 3D plot.

    Parameters:

    * ax: axes of a matplotlib figure.
    * xmin, xmax, ymin, ymax, zmin, zmax: floats
        Lower and upper limites along the x-, y- and z- axes.
    '''

    x = [xmin, xmax, xmin, xmin]
    y = [ymin, ymin, ymax, ymin]
    z = [zmin, zmin, zmin, zmax]
    ax.scatter(x, y, z, s=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
