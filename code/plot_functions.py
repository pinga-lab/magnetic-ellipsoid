import os
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np


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


def draw_main_system(ax, length_axes = 1., label_size = 22, elev=200, azim=-20):
    '''
    Plot the axes forming the main coordinate system.

    input
    ax: axes of a matplotlib figure.
    length_axes: float - length of the axes (in meters).
        Default is 1.
    label_size: float - size of the label font. Default is 22.
    elev and azim: floats - parameters controlling the view
        of the figure. Default is 200 and -20, respectively.

    output
    matplotlib objects
    '''

    # x-axis
    ax.quiver(length_axes, 0., 0., length_axes, 0., 0.,
              length = length_axes, color='k', linewidth=2.0, linestyle='-',
              arrow_length_ratio=0.1)
    ax.text(1.05*length_axes, 0., 0., '$x$', color='k', fontsize=label_size)

    # y-axis
    ax.quiver(0., length_axes, 0., 0., length_axes, 0.,
              length = length_axes, color='k', linewidth=2.0, linestyle='-',
              arrow_length_ratio=0.1)
    ax.text(0., 1.05*length_axes, 0., '$y$', color='k', fontsize=label_size)

    # z-axis
    ax.quiver(0., 0., length_axes, 0., 0., length_axes,
              length = length_axes, color='k', linewidth=2.0, linestyle='-',
              arrow_length_ratio=0.1)
    ax.text(0., 0., 1.05*length_axes, '$z$', color='k', fontsize=label_size)

    ax.axis('off')

    #ax.set_xlim(0., 1.1)
    #ax.set_ylim(0., 1.1)
    #ax.set_zlim(0., 1.1)

    ax.view_init(elev=elev, azim=azim)


def draw_ellipsoid(ax, xc, yc, zc, a, b, c, strike, dip, rake,
                   u, v, body_color, body_alpha,
                   plot_axes=True, axes_color=(0,0,0), label_size=16):
    '''
    Plot an ellipsoidal body with axes a, b, c and
    origin at (xc, yc, zc).

    input
    ax: axes of a matplotlib figure.
    xc: float - Cartesian coordinate x (in meters) of the origin
        referred to the main system.
    yc: float - Cartesian coordinate y (in meters) of the origin
        referred to the main system.
    zc: float - Cartesian coordinate z (in meters) of the origin
        referred to the main system.
    a: float - axis a (in meters)
    b: float - axis b (in meters)
    c: float - axis c (in meters)
    strike: float - angle strike (in degrees)
    dip: float - angle dip (in degrees)
    rake: float - angle rake (in degrees)
    u: None or numpy array 1D - angular spherical
        coordinates (in radians) for plotting the ellipsoidal surface.
    v: None or numpy array 1D - angular spherical
        coordinates (in radians) for plotting the ellipsoidal surface.
    body_color: RGB matplotlib tuple - color of the body.
    body_alpha: float - transparency of the body.
    plot_axes: boolean - If True (default), plot the body axes.
    axes_color: RGB matplotlib tuple - color of the axes. The default
        is (0,0,0) - black color.
    label_size: float - size of the label font. Default is 22.

    output
    matplotlib objects
    '''

    alpha, gamma, delta = structural_orientation(strike, dip, rake)

    if (a > b > c):
        V = V_triaxial(alpha, gamma, delta)

    if (a > b == c):
        V = V_prolate(alpha, gamma, delta)

    if (a < b == c):
        V = V_oblate(alpha, gamma, delta)

    if (a == b == c):
        V = np.identity(3)

    if plot_axes is True:

        ax.quiver(xc+V[0,0]*a, yc+V[1,0]*a, zc+V[2,0]*a, V[0,0], V[1,0], V[2,0],
                  length=a, color=axes_color, linewidth=3.0, linestyle='-',
                  arrow_length_ratio=0.1)
        ax.text(xc+V[0,0]*a*1.05, yc+V[1,0]*a*1.05, zc+V[2,0]*a*1.05,
                '$a \hat{\mathbf{v}}_{1}$', color=axes_color, fontsize=label_size)

        ax.quiver(xc+V[0,1]*b, yc+V[1,1]*b, zc+V[2,1]*b, V[0,1], V[1,1], V[2,1],
                  length=b, color=axes_color, linewidth=3.0, linestyle='-',
                  arrow_length_ratio=0.1)
        ax.text(xc+V[0,1]*b*1.05, yc+V[1,1]*b*1.05, zc+V[2,1]*b*1.05,
                '$b \hat{\mathbf{v}}_{2}$', color=axes_color, fontsize=label_size)

        ax.quiver(xc+V[0,2]*c, yc+V[1,2]*c, zc+V[2,2]*c, V[0,2], V[1,2], V[2,2],
                  length=c, color=axes_color, linewidth=3.0, linestyle='-',
                  arrow_length_ratio=0.1)
        ax.text(xc+V[0,2]*c*1.05, yc+V[1,2]*c*1.05, zc+V[2,2]*c*1.05,
                '$c \hat{\mathbf{v}}_{3}$', color=axes_color, fontsize=label_size)


    if (u is not None) and (v is not None):

        # Cartesian coordinates referred to the body system
        # (https://en.wikipedia.org/wiki/Ellipsoid)
        x1 = a * np.outer(np.cos(u), np.sin(v))
        x2 = b * np.outer(np.sin(u), np.sin(v))
        x3 = c * np.outer(np.ones_like(u), np.cos(v))

        # Cartesian coordinates referred to the main system
        x = V[0,0]*x1 + V[0,1]*x2 + V[0,2]*x3 + xc
        y = V[1,0]*x1 + V[1,1]*x2 + V[1,2]*x3 + yc
        z = V[2,0]*x1 + V[2,1]*x2 + V[2,2]*x3 + zc

        # Plot:
        ax.plot_surface(x, y, z, linewidth=0., color=body_color, alpha=body_alpha)


def V_triaxial(alpha, gamma, delta):
    '''
    Calculates the coordinate transformation matrix
    for a triaxial model.

    input
    alpha: float - angle alpha (in radians)
    gamma: float - angle gamma (in radians)
    delta: float - angle delta (in radians)

    output
    V: numpy array 2D - coordinate transformation matrix
    '''

    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)

    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)

    v1 = np.array([-cos_alpha*cos_delta,
                   -sin_alpha*cos_delta,
                   -sin_delta])

    v2 = np.array([ cos_alpha*cos_gamma*sin_delta + sin_alpha*sin_gamma,
                    sin_alpha*cos_gamma*sin_delta - cos_alpha*sin_gamma,
                   -cos_gamma*cos_delta])

    v3 = np.array([ sin_alpha*cos_gamma - cos_alpha*sin_gamma*sin_delta,
                   -cos_alpha*cos_gamma - sin_alpha*sin_gamma*sin_delta,
                    sin_gamma*cos_delta])

    V = np.vstack((v1, v2, v3)).T

    return V


def V_prolate(alpha, gamma, delta):
    '''
    Calculates the coordinate transformation matrix
    for a prolate model.

    input
    alpha: float - angle alpha (in radians)
    gamma: float - angle gamma (in radians)
    delta: float - angle delta (in radians)

    output
    V: numpy array 2D - coordinate transformation matrix
    '''

    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)

    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)

    v1 = np.array([-cos_alpha*cos_delta,
                   -sin_alpha*cos_delta,
                   -sin_delta])

    v2 = np.array([ cos_alpha*cos_gamma*sin_delta + sin_alpha*sin_gamma,
                    sin_alpha*cos_gamma*sin_delta - cos_alpha*sin_gamma,
                   -cos_gamma*cos_delta])

    v3 = np.array([ sin_alpha*cos_gamma - cos_alpha*sin_gamma*sin_delta,
                   -cos_alpha*cos_gamma - sin_alpha*sin_gamma*sin_delta,
                    sin_gamma*cos_delta])

    V = np.vstack((v1, v2, v3)).T

    return V


def V_oblate(alpha, gamma, delta):
    '''
    Calculates the coordinate transformation matrix
    for an oblate model.

    input
    alpha: float - angle alpha (in radians)
    gamma: float - angle gamma (in radians)
    delta: float - angle delta (in radians)

    output
    V: numpy array 2D - coordinate transformation matrix
    '''

    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)

    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)

    v1 = np.array([-cos_alpha*sin_gamma*sin_delta + sin_alpha*cos_gamma,
                   -sin_alpha*sin_gamma*sin_delta - cos_alpha*cos_gamma,
                   sin_gamma*cos_delta])

    v2 = np.array([-cos_alpha*cos_delta,
                   -sin_alpha*cos_delta,
                   -sin_delta])

    v3 = np.array([sin_alpha*sin_gamma + cos_alpha*cos_gamma*sin_delta,
                   -cos_alpha*sin_gamma + sin_alpha*cos_gamma*sin_delta,
                   -cos_gamma*cos_delta])

    V = np.vstack((v1, v2, v3)).T

    return V


def structural_orientation(strike, dip, rake):
    '''
    Calculates the angles that describe how a line is
    positioned in space according to structural geology.
    The computation is defined acording to Clark (1986)
    and Allmendinger et al. (2012).

    input:
    strike: float - angle strike (in degrees)
    dip: float - angle dip (in degrees)
    rake: float - angle rake (in degrees)

    output:
    alpha: float - angle alpha (in radians)
    gamma: float - angle gamma (in radians)
    delta: float - angle delta (in radians)

    References:

    Allmendinger, R., Cardozo, N., and Fisher, D. M.:
    Structural geology algorithms : vectors and tensors,
    Cambridge University Press, 2012.

    Clark, D., Saul, S., and Emerson, D.: Magnetic and
    gravity anomalies of a triaxial ellipsoid, Exploration
    Geophysics, 17, 189-200, 1986.
    '''

    strike_r = np.deg2rad(strike)
    cos_dip = np.cos(np.deg2rad(dip))
    sin_dip = np.sin(np.deg2rad(dip))
    cos_rake = np.cos(np.deg2rad(rake))
    sin_rake = np.sin(np.deg2rad(rake))

    aux = sin_dip*sin_rake
    aux1 = cos_rake/np.sqrt(1 - aux*aux)

    if aux1 > 1.:
        aux1 = 1.
    if aux1 < -1.:
        aux1 = -1.

    alpha = strike_r - np.arccos(aux1)
    gamma = -np.arctan(cos_dip/(sin_dip*cos_rake))
    delta = np.arcsin(aux)

    assert delta <= np.pi/2, 'delta must be lower than \
or equalt to 90 degrees'

    assert (gamma >= -np.pi/2) and (gamma <= np.pi/2), 'gamma must lie between \
-90 and 90 degrees.'

    return alpha, gamma, delta
