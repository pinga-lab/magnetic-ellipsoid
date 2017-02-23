r"""
The potential fields of a homogeneous triaxial ellipsoid.
"""
from __future__ import division

import numpy as np
from scipy.special import ellipeinc, ellipkinc

from ..constants import SI2MGAL, G, CM, T2NT, SI2EOTVOS, PERM_FREE_SPACE
from .. import utils
from .._our_duecredit import due, Doi


due.cite(Doi("XXXXXXXXXXXXXXXXX"),
         description='Forward modeling formula for triaxial ellipsoids.',
         path='fatiando.gravmag.triaxial_ellipsoid')


def tf(xp, yp, zp, ellipsoids, F, inc, dec, dmag=True, pmag=None):
    r"""
    The total-field anomaly produced by triaxial ellipsoids
    (Clark et al., 1986).

    .. math::

        \Delta T = |\mathbf{T}| - |\mathbf{F}|,

    where :math:`\mathbf{T}` is the measured field and :math:`\mathbf{F}` is
    the local-geomagnetic field.

    The anomaly of a homogeneous ellipsoid can be calculated as:

    .. math::

        \Delta T \approx \hat{\mathbf{F}}\cdot\mathbf{B}.

    where :math:`\mathbf{B}` is the magnetic induction produced by the
    ellipsoid.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoids : list of :class:`fatiando.mesher.EllipsoidTriaxial`
        The ellipsoids. Ellipsoids must have the physical property
        ``'k'`` and/or ``'remanence'``. Ellipsoids that are ``None`` or
        without ``'k'`` and ``'remanence'`` will be ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * dmag : boolean
        If True, will include the self-demagnetization.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of the resultant magnetization of the ellipsoids. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * tf : array
        The total-field anomaly

    References:

    Clark, D. A., S. J. Saul and D. W. Emerson (1986),
    Magnetic and gravity anomalies of a triaxial ellipsoid.

    """
    fx, fy, fz = utils.dircos(inc, dec)
    Bx = bx(xp, yp, zp, ellipsoids, F, inc, dec, dmag, pmag)
    By = by(xp, yp, zp, ellipsoids, F, inc, dec, dmag, pmag)
    Bz = bz(xp, yp, zp, ellipsoids, F, inc, dec, dmag, pmag)

    return fx*Bx + fy*By + fz*Bz


def bx(xp, yp, zp, ellipsoids, F, inc, dec, dmag=True, pmag=None):
    r"""
    The x component of the magnetic induction produced by triaxial
    ellipsoids (Clark et al., 1986).

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoids : list of :class:`fatiando.mesher.EllipsoidTriaxial`
        The ellipsoids. Ellipsoids must have the physical property
        ``'k'`` and/or ``'remanence'``. Ellipsoids that are ``None`` or
        without ``'k'`` and ``'remanence'`` will be ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * dmag : boolean
        If True, will include the self-demagnetization.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of the resultant magnetization of the ellipsoids. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * bx: array
        The x component of the magnetic induction

    References:

    Clark, D. A., S. J. Saul and D. W. Emerson (1986),
    Magnetic and gravity anomalies of a triaxial ellipsoid.

    """

    res = 0
    for ellipsoid in ellipsoids:
        if ellipsoid is None:
            continue
        if 'k' not in ellipsoid.props and 'remanence' not in ellipsoid.props:
            continue
        alpha, gamma, delta = structural_angles(ellipsoid.strike,
                                                ellipsoid.dip, ellipsoid.rake)
        matrix = V(alpha, gamma, delta)
        b1 = _bx(xp, yp, zp, ellipsoid, F, inc, dec, dmag, pmag)
        b2 = _by(xp, yp, zp, ellipsoid, F, inc, dec, dmag, pmag)
        b3 = _bz(xp, yp, zp, ellipsoid, F, inc, dec, dmag, pmag)
        res += matrix[0, 0]*b1 + matrix[0, 1]*b2 + matrix[0, 2]*b3

    return res


def by(xp, yp, zp, ellipsoids, F, inc, dec, dmag=True, pmag=None):
    r"""
    The y component of the magnetic induction produced by triaxial
    ellipsoids (Clark et al., 1986).

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoids : list of :class:`fatiando.mesher.EllipsoidTriaxial`
        The ellipsoids. Ellipsoids must have the physical property
        ``'k'`` and/or ``'remanence'``. Ellipsoids that are ``None`` or
        without ``'k'`` and ``'remanence'`` will be ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * dmag : boolean
        If True, will include the self-demagnetization.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of the resultant magnetization of the ellipsoids. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * by: array
        The y component of the magnetic induction

    References:

    Clark, D. A., S. J. Saul and D. W. Emerson (1986),
    Magnetic and gravity anomalies of a triaxial ellipsoid.

    """

    res = 0
    for ellipsoid in ellipsoids:
        if ellipsoid is None:
            continue
        if 'k' not in ellipsoid.props and 'remanence' not in ellipsoid.props:
            continue
        alpha, gamma, delta = structural_angles(ellipsoid.strike,
                                                ellipsoid.dip, ellipsoid.rake)
        matrix = V(alpha, gamma, delta)
        b1 = _bx(xp, yp, zp, ellipsoid, F, inc, dec, dmag, pmag)
        b2 = _by(xp, yp, zp, ellipsoid, F, inc, dec, dmag, pmag)
        b3 = _bz(xp, yp, zp, ellipsoid, F, inc, dec, dmag, pmag)
        res += matrix[1, 0]*b1 + matrix[1, 1]*b2 + matrix[1, 2]*b3

    return res


def bz(xp, yp, zp, ellipsoids, F, inc, dec, dmag=True, pmag=None):
    r"""
    The z component of the magnetic induction produced by triaxial
    ellipsoids (Clark et al., 1986).

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoids : list of :class:`fatiando.mesher.EllipsoidTriaxial`
        The ellipsoids. Ellipsoids must have the physical property
        ``'k'`` and/or ``'remanence'``. Ellipsoids that are ``None`` or
        without ``'k'`` and ``'remanence'`` will be ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * dmag : boolean
        If True, will include the self-demagnetization.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of the resultant magnetization of the ellipsoids. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * bz: array
        The z component of the magnetic induction

    References:

    Clark, D. A., S. J. Saul and D. W. Emerson (1986),
    Magnetic and gravity anomalies of a triaxial ellipsoid.

    """

    res = 0
    for ellipsoid in ellipsoids:
        if ellipsoid is None:
            continue
        if 'k' not in ellipsoid.props and 'remanence' not in ellipsoid.props:
            continue
        alpha, gamma, delta = structural_angles(ellipsoid.strike,
                                                ellipsoid.dip, ellipsoid.rake)
        matrix = V(alpha, gamma, delta)
        b1 = _bx(xp, yp, zp, ellipsoid, F, inc, dec, dmag, pmag)
        b2 = _by(xp, yp, zp, ellipsoid, F, inc, dec, dmag, pmag)
        b3 = _bz(xp, yp, zp, ellipsoid, F, inc, dec, dmag, pmag)
        res += matrix[2, 0]*b1 + matrix[2, 1]*b2 + matrix[2, 2]*b3

    return res


def _bx(xp, yp, zp, ellipsoid, F, inc, dec, dmag=True, pmag=None):
    r"""
    The x component of the magnetic induction produced by triaxial
    ellipsoids in the ellipsoid system (Clark et al., 1986).

    The coordinate system of the input parameters is x -> semi-axis a,
    y -> semi-axis b and z -> semi-axis c.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoid : element of :class:`fatiando.mesher.EllipsoidTriaxial`
        The ellipsoid. The ellipsoid must have the physical property
        ``'k'`` and/or ``'remanence'``. If the ellipsoids is ``None`` or
        without ``'k'`` and ``'remanence'``, it is ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * dmag : boolean
        If True, include the self-demagnetization.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of the resultant magnetization of the ellipsoid. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * bx : array
        The x component of the magnetic induction in the ellipsoid system.

    References:

    Clark, D. A., S. J. Saul and D. W. Emerson (1986),
    Magnetic and gravity anomalies of a triaxial ellipsoid.

    """

    alpha, gamma, delta = structural_angles(ellipsoid.strike, ellipsoid.dip,
                                            ellipsoid.rake)
    matrix = V(alpha, gamma, delta)

    if pmag is None:
        k1, k2, k3, alpha_susc, gamma_susc, delta_susc = ellipsoid.props['k']
        suscep = K(k1, k2, k3, alpha_susc, gamma_susc, delta_susc)
        if 'remanence' in ellipsoid.props:
            Hr, incr, decr = ellipsoid.props['remanence']
        else:
            Hr, incr, decr = 0, 0, 0
        if dmag is True:
            axes = [ellipsoid.a, ellipsoid.b, ellipsoid.c]
            mx, my, mz = magnetization(suscep, F, inc, dec,
                                       Hr, incr, decr, matrix, axes)
        else:
            mx, my, mz = magnetization(suscep, F, inc, dec,
                                       Hr, incr, decr, matrix, axes=None)
    else:
        mx, my, mz = pmag

    x1, x2, x3 = x1x2x3(xp, yp, zp, ellipsoid.x, ellipsoid.y, ellipsoid.z,
                        matrix)
    lamb = _lamb(x1, x2, x3, ellipsoid.a, ellipsoid.b, ellipsoid.c)
    denominator = _dlamb_aux(x1, x2, x3,
                             ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb)
    dlamb = _dlamb(x1, x2, x3, ellipsoid.a, ellipsoid.b, ellipsoid.c,
                   lamb, denominator, deriv='x')
    h1 = _hv(ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb, v='x')
    h2 = _hv(ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb, v='y')
    h3 = _hv(ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb, v='z')
    kappa, phi = _E_F_field_args(ellipsoid.a, ellipsoid.b, ellipsoid.c,
                                 lamb)
    g = _gv_tejedor(ellipsoid.a, ellipsoid.b, ellipsoid.c,
                    kappa, phi, lamb, v='x')
    res = dlamb*(h1*x1*mx + h2*x2*my + h3*x3*mz) + g*mx
    res *= -CM*T2NT*2*np.pi*ellipsoid.a*ellipsoid.b*ellipsoid.c
    return res


def _by(xp, yp, zp, ellipsoid, F, inc, dec, dmag=True, pmag=None):
    r"""
    The y component of the magnetic induction produced by triaxial
    ellipsoids in the ellipsoid system (Clark et al., 1986).

    The coordinate system of the input parameters is x -> semi-axis a,
    y -> semi-axis b and z -> semi-axis c.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoid : element of :class:`fatiando.mesher.EllipsoidTriaxial`
        The ellipsoid. The ellipsoid must have the physical property
        ``'k'`` and/or ``'remanence'``. If the ellipsoids is ``None`` or
        without ``'k'`` and ``'remanence'``, it is ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * dmag : boolean
        If True, include the self-demagnetization.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of the resultant magnetization of the ellipsoid. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * by : array
        The y component of the magnetic induction in the ellipsoid system.

    References:

    Clark, D. A., S. J. Saul and D. W. Emerson (1986),
    Magnetic and gravity anomalies of a triaxial ellipsoid.

    """

    alpha, gamma, delta = structural_angles(ellipsoid.strike, ellipsoid.dip,
                                            ellipsoid.rake)
    matrix = V(alpha, gamma, delta)

    if pmag is None:
        k1, k2, k3, alpha_susc, gamma_susc, delta_susc = ellipsoid.props['k']
        suscep = K(k1, k2, k3, alpha_susc, gamma_susc, delta_susc)
        if 'remanence' in ellipsoid.props:
            Hr, incr, decr = ellipsoid.props['remanence']
        else:
            Hr, incr, decr = 0, 0, 0
        if dmag is True:
            axes = [ellipsoid.a, ellipsoid.b, ellipsoid.c]
            mx, my, mz = magnetization(suscep, F, inc, dec,
                                       Hr, incr, decr, matrix, axes)
        else:
            mx, my, mz = magnetization(suscep, F, inc, dec,
                                       Hr, incr, decr, matrix, axes=None)
    else:
        mx, my, mz = pmag

    x1, x2, x3 = x1x2x3(xp, yp, zp, ellipsoid.x, ellipsoid.y, ellipsoid.z,
                        matrix)
    lamb = _lamb(x1, x2, x3, ellipsoid.a, ellipsoid.b, ellipsoid.c)
    denominator = _dlamb_aux(x1, x2, x3,
                             ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb)
    dlamb = _dlamb(x1, x2, x3, ellipsoid.a, ellipsoid.b, ellipsoid.c,
                   lamb, denominator, deriv='y')
    h1 = _hv(ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb, v='x')
    h2 = _hv(ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb, v='y')
    h3 = _hv(ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb, v='z')
    kappa, phi = _E_F_field_args(ellipsoid.a, ellipsoid.b, ellipsoid.c,
                                 lamb)
    g = _gv_tejedor(ellipsoid.a, ellipsoid.b, ellipsoid.c,
                    kappa, phi, lamb, v='y')
    res = dlamb*(h1*x1*mx + h2*x2*my + h3*x3*mz) + g*my
    res *= -CM*T2NT*2*np.pi*ellipsoid.a*ellipsoid.b*ellipsoid.c
    return res


def _bz(xp, yp, zp, ellipsoid, F, inc, dec, dmag=True, pmag=None):
    r"""
    The z component of the magnetic induction produced by triaxial
    ellipsoids in the ellipsoid system (Clark et al., 1986).

    The coordinate system of the input parameters is x -> semi-axis a,
    y -> semi-axis b and z -> semi-axis c.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoid : element of :class:`fatiando.mesher.EllipsoidTriaxial`
        The ellipsoid. The ellipsoid must have the physical property
        ``'k'`` and/or ``'remanence'``. If the ellipsoids is ``None`` or
        without ``'k'`` and ``'remanence'``, it is ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * dmag : boolean
        If True, will include the self-demagnetization.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of the resultant magnetization of the ellipsoid. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * bz : array
        The z component of the magnetic induction in the ellipsoid system.

    References:

    Clark, D. A., S. J. Saul and D. W. Emerson (1986),
    Magnetic and gravity anomalies of a triaxial ellipsoid.

    """

    alpha, gamma, delta = structural_angles(ellipsoid.strike, ellipsoid.dip,
                                            ellipsoid.rake)
    matrix = V(alpha, gamma, delta)

    if pmag is None:
        k1, k2, k3, alpha_susc, gamma_susc, delta_susc = ellipsoid.props['k']
        suscep = K(k1, k2, k3, alpha_susc, gamma_susc, delta_susc)
        if 'remanence' in ellipsoid.props:
            Hr, incr, decr = ellipsoid.props['remanence']
        else:
            Hr, incr, decr = 0, 0, 0
        if dmag is True:
            axes = [ellipsoid.a, ellipsoid.b, ellipsoid.c]
            mx, my, mz = magnetization(suscep, F, inc, dec,
                                       Hr, incr, decr, matrix, axes)
        else:
            mx, my, mz = magnetization(suscep, F, inc, dec,
                                       Hr, incr, decr, matrix, axes=None)
    else:
        mx, my, mz = pmag

    x1, x2, x3 = x1x2x3(xp, yp, zp, ellipsoid.x, ellipsoid.y, ellipsoid.z,
                        matrix)
    lamb = _lamb(x1, x2, x3, ellipsoid.a, ellipsoid.b, ellipsoid.c)
    denominator = _dlamb_aux(x1, x2, x3,
                             ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb)
    dlamb = _dlamb(x1, x2, x3, ellipsoid.a, ellipsoid.b, ellipsoid.c,
                   lamb, denominator, deriv='z')
    h1 = _hv(ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb, v='x')
    h2 = _hv(ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb, v='y')
    h3 = _hv(ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb, v='z')
    kappa, phi = _E_F_field_args(ellipsoid.a, ellipsoid.b, ellipsoid.c,
                                 lamb)
    g = _gv_tejedor(ellipsoid.a, ellipsoid.b, ellipsoid.c,
                    kappa, phi, lamb, v='z')
    res = dlamb*(h1*x1*mx + h2*x2*my + h3*x3*mz) + g*mz
    res *= -CM*T2NT*2*np.pi*ellipsoid.a*ellipsoid.b*ellipsoid.c
    return res


def structural_angles(strike, dip, rake):
    '''
    Calculates the orientation angles alpha, gamma
    and delta (Clark et al., 1986)
    as functions of the geological angles strike, dip and
    rake (Clark et al., 1986; Allmendinger et al., 2012).
    The function implements the formulas presented by
    Clark et al. (1986).

    Parameters:

    *strike: float
             strike direction (in degrees).
    *dip: float
          true dip (in degrees).
    *rake: float
           angle between the strike and the semi-axis a
           of the body (in degrees).

    Returns:

    *alpha, gamma, delta: float, float, float
            orientation angles (in radians) defined according
            to Clark et al. (1986).

    References:

    Clark, D., Saul, S., and Emerson, D.: Magnetic and gravity anomalies
    of a triaxial ellipsoid, Exploration Geophysics, 17, 189-200, 1986.

    Allmendinger, R., Cardozo, N., and Fisher, D. M.:
    Structural geology algorithms : vectors and tensors,
    Cambridge University Press, 2012.
    '''

    strike_r = np.deg2rad(strike)
    cos_dip = np.cos(np.deg2rad(dip))
    sin_dip = np.sin(np.deg2rad(dip))
    cos_rake = np.cos(np.deg2rad(rake))
    sin_rake = np.sin(np.deg2rad(rake))

    aux = sin_dip*sin_rake
    aux1 = cos_rake/np.sqrt(1 - aux*aux)
    aux2 = sin_dip*cos_rake

    if aux1 > 1.:
        aux1 = 1.
    if aux1 < -1.:
        aux1 = -1.

    alpha = strike_r - np.arccos(aux1)
    if aux2 != 0:
        gamma = -np.arctan(cos_dip/aux2)
    else:
        if cos_dip >= 0:
            gamma = np.pi/2
        if cos_dip <= 0:
            gamma = -np.pi/2
    delta = np.arcsin(aux)

    assert delta <= np.pi/2, 'delta must be lower than \
or equalt to 90 degrees'

    assert (gamma >= -np.pi/2) and (gamma <= np.pi/2), 'gamma must lie between \
-90 and 90 degrees.'

    return alpha, gamma, delta


def V(alpha, gamma, delta):
    '''
    Calculates the coordinate transformation matrix
    for a triaxial model. The columns of this matrix
    are defined according to the unit vectors v1, v2
    and v3 presented by Clark et al. (1986, p. 192).

    Parameters:

    *alpha: float
            Orientation angle (in radians).
    *gamma: float
            Orientation angle (in radians).
    *delta: float
            Orientation angle (in radians).

    Returns:

    *matrix: numpy array 2D
             A 3x3 matrix with the direction cosines of the new coordinate
             system.

    References:

    Clark, D., Saul, S., and Emerson, D.: Magnetic and gravity anomalies
    of a triaxial ellipsoid, Exploration Geophysics, 17, 189-200, 1986.
    '''

    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)

    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)

    v1 = np.array([-cos_alpha*cos_delta, -sin_alpha*cos_delta, -sin_delta])

    v2 = np.array([cos_alpha*cos_gamma*sin_delta + sin_alpha*sin_gamma,
                   sin_alpha*cos_gamma*sin_delta - cos_alpha*sin_gamma,
                   -cos_gamma*cos_delta])

    v3 = np.array([sin_alpha*cos_gamma - cos_alpha*sin_gamma*sin_delta,
                   -cos_alpha*cos_gamma - sin_alpha*sin_gamma*sin_delta,
                   sin_gamma*cos_delta])

    matrix = np.vstack((v1, v2, v3)).T

    return matrix


def x1x2x3(xp, yp, zp, xc, yc, zc, matrix):
    '''
    Calculates the x, y and z coordinates referred to the
    ellipsoid coordinate system.

    input
    xp: numpy array 1D - x coordinates in the main system (in meters).
    yp: numpy array 1D - y coordinates in the main system (in meters).
    zp: numpy array 1D - z coordinates in the main system (in meters).
    xc: float - x coordinate of the ellipsoid center in the main
                system (in meters).
    yc: float - y coordinate of the ellipsoid center in the main
                system (in meters).
    zc: float - z coordinate of the ellipsoid center in the main
                system (in meters).
    matrix: numpy array 2D - coordinate transformation matrix.

    output
    x1: numpy array 1D - x coordinates in the ellipsoid system (in meters).
    x2: numpy array 1D - y coordinates in the ellipsoid system (in meters).
    x3: numpy array 1D - z coordinates in the ellipsoid system (in meters).
    '''

    assert xp.size == yp.size == zp.size, \
        'xp, yp and zp must have the same size'

    assert np.allclose(np.dot(matrix.T, matrix), np.identity(3)), \
        'matrix must be a valid coordinate transformation matrix'

    x1 = matrix[0, 0]*(xp - xc) + matrix[1, 0]*(yp - yc) + \
        matrix[2, 0]*(zp - zc)
    x2 = matrix[0, 1]*(xp - xc) + matrix[1, 1]*(yp - yc) + \
        matrix[2, 1]*(zp - zc)
    x3 = matrix[0, 2]*(xp - xc) + matrix[1, 2]*(yp - yc) + \
        matrix[2, 2]*(zp - zc)

    return x1, x2, x3


def K(k1, k2, k3, alpha, gamma, delta):
    '''
    Calculates the susceptibility tensor (in SI) in the main system.

    input
    k1: float - maximum eigenvalue of the susceptibility matrix K.
    k2: float - intermediate eigenvalue of the susceptibility matrix K.
    k3: float - minimum eigenvalue of the susceptibility matrix K.
    alpha: float - orientation angle (in degrees) defining the
    susceptibility tensor.
    gamma: float - orientation angle (in degrees) defining the
    susceptibility tensor.
    delta: float - orientation angle (in degrees) defining the
    susceptibility tensor.

    output
    K: numpy array 2D - susceptibility tensor in the main system (in SI).
    '''

    assert k1 >= k2 >= k3, 'k1, k2 and k3 must be the maximum, \
        intermediate and minimum eigenvalues'

    assert (k1 > 0) and (k2 > 0) and (k3 > 0), 'k1, k2 and k3 must \
        be all positive'

    U = V(alpha, gamma, delta)

    K = np.dot(U, np.diag([k1, k2, k3]))
    K = np.dot(K, U.T)

    return K


def _lamb(x, y, z, a, b, c):
    '''
    Calculates the parameter lambda.

    input
    x: numpy array 1D - x coordinates in the ellipsoid system (in meters).
    y: numpy array 1D - y coordinates in the ellipsoid system (in meters).
    z: numpy array 1D - z coordinates in the ellipsoid system (in meters).
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    c: float - semi-axis c (in meters).

    output
    lamb: numpy array 1D - parameter lambda for each point in the
        ellipsoid system.
    '''

    assert a > b > c, 'a must be greater than b and b must be greater than c'

    assert (a > 0) and (b > 0) and (c > 0), 'a, b and c must \
        be all positive'

    # auxiliary variables
    p2 = a*a + b*b + c*c - x*x - y*y - z*z
    p1 = (b*c*b*c) + (a*c*a*c) + (a*b*a*b) - (b*b + c*c)*(x*x) \
        - (a*a + c*c)*(y*y) - (a*a + b*b)*(z*z)
    p0 = (a*b*c*a*b*c) - (b*c*x*b*c*x) - (a*c*y*a*c*y) - (a*b*z*a*b*z)
    Q = (3.*p1 - p2*p2)/9.
    R = (9.*p1*p2 - 27.*p0 - 2.*p2*p2*p2)/54.

    p3 = R/np.sqrt(-(Q*Q*Q))

    assert np.alltrue(p3 <= 1.), 'arccos argument greater than 1'

    assert np.alltrue(Q*Q*Q + R*R < 0), 'the polynomial discriminant \
        must be negative'

    theta = np.arccos(p3)

    lamb = 2.*np.sqrt(-Q)*np.cos(theta/3.) - p2/3.

    return lamb


def _cubic_coeffs(x, y, z, a, b, c):
    '''
    Calculates the coefficients of the cubic equation defining
    a triaxial ellipsoid.

    input
    x: numpy array 1D - x coordinates in the ellipsoid system (in meters).
    y: numpy array 1D - y coordinates in the ellipsoid system (in meters).
    z: numpy array 1D - z coordinates in the ellipsoid system (in meters).
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    c: float - semi-axis c (in meters).

    output
    p2: numpy array 1D - coefficient multiplying the quadratic term.
    p1: numpy array 1D - coefficient multiplying the linear term.
    p0: numpy array 1D - coefficient multiplying the constant.
    '''

    assert a > b > c, 'a must be greater than b and b must be greater than c'

    assert (a > 0) and (b > 0) and (c > 0), 'a, b and c must \
        be all positive'

    # auxiliary variables
    p2 = a*a + b*b + c*c - x*x - y*y - z*z
    p1 = (b*c*b*c) + (a*c*a*c) + (a*b*a*b) - (b*b + c*c)*(x*x) \
        - (a*a + c*c)*(y*y) - (a*a + b*b)*(z*z)
    p0 = (a*b*c*a*b*c) - (b*c*x*b*c*x) - (a*c*y*a*c*y) - (a*b*z*a*b*z)

    return p2, p1, p0


def _dlamb(x, y, z, a, b, c, lamb, denominator, deriv='x'):
    '''
    Calculates the spatial derivative of the parameter lambda
    with respect to the coordinates x, y or z in the ellipsoid system.

    input
    x: numpy array 1D - x coordinates in the ellipsoid system (in meters).
    y: numpy array 1D - y coordinates in the ellipsoid system (in meters).
    z: numpy array 1D - z coordinates in the ellipsoid system (in meters).
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    c: float - semi-axis c (in meters).
    lambda: numpy array 1D - parameter lambda defining the surface of the
        triaxial ellipsoid.
    denominator: numpy array 1D - denominator of the equation used to
        calculate the spatial derivative of lambda.
    deriv: string - defines the coordinate with respect to which the
        derivative will be calculated. It must be 'x', 'y' or 'z'.

    output
    dlamb_dv: numpy array 1D - derivative of lambda with respect to the
        coordinate v = x, y, z in the ellipsoid system.
    '''

    assert a > b > c, 'a must be greater than b and b must be greater than c'

    assert (a > 0) and (b > 0) and (c > 0), 'a, b and c must \
        be all positive'

    assert deriv in ['x', 'y', 'z'], 'deriv must represent a coordinate \
        x, y or z'

    assert denominator.size == lamb.size == x.size == y.size == z.size, \
        'x, y, z, lamb and denominator must have the same size'

    if deriv is 'x':
        dlamb_dv = (2*x/(a*a + lamb))/denominator

    if deriv is 'y':
        dlamb_dv = (2*y/(b*b + lamb))/denominator

    if deriv is 'z':
        dlamb_dv = (2*z/(c*c + lamb))/denominator

    return dlamb_dv


def _dlamb_aux(x, y, z, a, b, c, lamb):
    '''
    Calculates an auxiliary variable used to calculate the spatial
    derivatives of the parameter lambda with respect to the
    coordinates x, y and z in the ellipsoid system.

    input
    x: numpy array 1D - x coordinates in the ellipsoid system (in meters).
    y: numpy array 1D - y coordinates in the ellipsoid system (in meters).
    z: numpy array 1D - z coordinates in the ellipsoid system (in meters).
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    c: float - semi-axis c (in meters).
    lambda: float - parameter lambda defining the surface of the triaxial
        ellipsoid.

    output
    aux: numpy array 1D - denominator of the expression used to calculate
        the spatial derivative of lambda.
    '''

    assert a > b > c, 'a must be greater than b and b must be greater than c'

    assert (a > 0) and (b > 0) and (c > 0), 'a, b and c must \
        be all positive'

    aux1 = x/(a*a + lamb)
    aux2 = y/(b*b + lamb)
    aux3 = z/(c*c + lamb)
    aux = aux1*aux1 + aux2*aux2 + aux3*aux3

    return aux


def _E_F_demag(a, b, c):
    '''
    Calculates the Legendre's normal elliptic integrals of first
    and second kinds which are used to calculate the demagnetizing
    factors.

    input:
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    c: float - semi-axis c (in meters).

    output:
    F - Legendre's normal elliptic integrals of first kind.
    E - Legendre's normal elliptic integrals of second kind.
    '''

    assert a > b > c, 'a must be greater than b and b must be greater than c'

    assert (a > 0) and (b > 0) and (c > 0), 'a, b and c must \
        be all positive'

    kappa = np.sqrt(((a*a-b*b)/(a*a-c*c)))
    phi = np.arccos(c/a)

    E = ellipeinc(phi, kappa*kappa)
    F = ellipkinc(phi, kappa*kappa)

    return E, F


def demag_factors(a, b, c):
    '''
    Calculates the demagnetizing factors n11, n22 and n33.

    input
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    c: float - semi-axis c (in meters).

    output
    n11: float - demagnetizing factor along the semi-axis a (in SI).
    n22: float - demagnetizing factor along the semi-axis b (in SI).
    n33: float - demagnetizing factor along the semi-axis c (in SI).
    '''

    assert a > b > c, 'a must be greater than b and b must be greater than c'

    assert (a > 0) and (b > 0) and (c > 0), 'a, b and c must \
        be all positive'

    E, F = _E_F_demag(a, b, c)

    aux1 = (a*b*c)/np.sqrt((a*a - c*c))
    n11 = (aux1/(a*a - b*b))*(F - E)
    n22 = -n11 + (aux1/(b*b - c*c))*E - (c*c)/(b*b - c*c)
    n33 = -(aux1/(b*b - c*c))*E + (b*b)/(b*b - c*c)

    return n11, n22, n33


def magnetization(suscep, F, inc, dec, RM, incrm, decrm, matrix, axes):
    '''
    Calculates the resultant magnetization corrected from
    demagnetizing in the main system.

    input
    suscep: numpy array 2D - susceptibility tensor in the main system
        (in SI).
    F: float - intensity of the local-geomagnetic field (in nT).
    inc: float - inclination of the local-geomagnetic field (in degrees)
        in the main coordinate system.
    dec: float - declination of the local-geomagnetic field (in degrees)
        in the main coordinate system.
    RM: float - intensity of the remanent magnetization (in A/m).
    incrm: float - inclination of the remanent magnetization (in degrees)
        in the main coordinate system.
    decrm: float - declination of the remanent magnetization (in degrees)
        in the main coordinate system.
    matrix: numpy array 2D - coordinate transformation matrix.
    axes: [a, b, c] or None - If None, it does not include the
        desmagnetization.

    output
    m: numpy array 1D - resultant magnetization (in A/m) in the
        main system.
    '''

    assert np.allclose(np.dot(matrix.T, matrix), np.identity(3)), \
        'matrix must be a valid coordinate transformation matrix'

    assert np.allclose(suscep.T, suscep), 'the susceptibility is a \
symmetrical tensor'

    if axes is not None:

        assert len(axes) == 3, 'axes must be a list containing the semi-axes'

        assert axes[0] > axes[1] > axes[2], 'axes must contain valid \
semi-axes'
        n11, n22, n33 = demag_factors(axes[0], axes[1], axes[2])

        assert n11 <= n22 <= n33, 'n11 must be smaller than n22 and \
n22 must be smaller than n33'

        assert (n11 >= 0) and (n22 >= 0) and (n33 >= 0), 'n11, n22 and n33 must \
be all positive or zero (for neglecting the self-demagnetization)'

        suscep_tilde = np.dot(np.dot(matrix.T, suscep), matrix)
        aux = np.linalg.inv(np.identity(3) + np.dot(suscep_tilde,
                                                    np.diag([n11, n22, n33])))
        Lambda = np.dot(np.dot(matrix, aux), matrix.T)

    else:
        Lambda = np.identity(3)

    H0 = utils.ang2vec(F/(4*np.pi*100), inc, dec)
    RM_vec = utils.ang2vec(RM, incrm, decrm)

    # resultant magnetization in the main system
    M = np.dot(Lambda, np.dot(suscep, H0) + RM_vec)

    return M


def _E_F_field(a, b, c, kappa, phi):
    '''
    Calculates the Legendre's normal elliptic integrals of first
    and second kinds which are used to calculate the potential
    fields outside the body.

    input:
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    c: float - semi-axis c (in meters).
    kappa: float - an argument of the elliptic integrals.
    phi: numpy array 1D - an argument of the elliptic integrals.

    output:
    F: numpy array 1D - Legendre's normal elliptic integrals of first kind.
    E: numpy array 1D - Legendre's normal elliptic integrals of second kind.
    '''

    assert a > b > c, 'a must be greater than b and b must be greater than c'

    assert (a > 0) and (b > 0) and (c > 0), 'a, b and c must \
        be all positive'

    E = ellipeinc(phi, kappa*kappa)
    F = ellipkinc(phi, kappa*kappa)

    return E, F


def _E_F_field_args(a, b, c, lamb):
    '''
    Calculates the arguments of the elliptic integrals defining
    the elements of the depolarization tensor without the body.

    input
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    c: float - semi-axis c (in meters).
    lambda: float - parameter lambda defining the surface of the triaxial
        ellipsoid.

    output
    kappa: numpy array 1D - parameter of the elliptic integral.
    phi: numpy array 1D - amplitude of the elliptic integral.
    '''

    assert a > b > c, 'a must be greater than b and b must be greater than c'

    assert (a > 0) and (b > 0) and (c > 0), 'a, b and c must \
        be all positive'

    kappa = np.sqrt((a*a - b*b)/(a*a - c*c))
    phi = np.arcsin(np.sqrt((a*a - c*c)/(a*a + lamb)))

    return kappa, phi


def _hv(a, b, c, lamb, v='x'):
    '''
    Calculates an auxiliary variable used to calculate the
    depolarization tensor outside the ellipsoidal body.

    input
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    c: float - semi-axis c (in meters).
    lambda: float - parameter lambda defining the surface of the triaxial
        ellipsoid.
    v: string - defines the coordinate with respect to which the
        variable hv will be calculated. It must be 'x', 'y' or 'z'.

    output
    hv: numpy array 1D - auxiliary variable.
    '''

    assert a > b > c, 'a must be greater than b and b must be greater than c'

    assert (a > 0) and (b > 0) and (c > 0), 'a, b and c must \
        be all positive'

    assert v in ['x', 'y', 'z'], "v must be 'x', 'y' or 'z'"

    aux1 = a*a + lamb
    aux2 = b*b + lamb
    aux3 = c*c + lamb
    R = np.sqrt(aux1*aux2*aux3)

    if v is 'x':
        hv = -1./(aux1*R)

    if v is 'y':
        hv = -1./(aux2*R)

    if v is 'z':
        hv = -1./(aux3*R)

    return hv


def _gv(a, b, c, kappa, phi, v='x'):
    '''
    Diagonal terms of the depolarization tensor defined outside the
    ellipsoidal body. These terms depend on the Legendre's normal
    elliptic integrals of first and second kinds (Clark, 1986).

    input
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    c: float - semi-axis c (in meters).
    kappa: numpy array 1D - parameter of the elliptic integral.
    phi: numpy array 1D - amplitude of the elliptic integral.
    v: string - defines the coordinate with respect to which the
        variable gv will be calculated. It must be 'x', 'y' or 'z'.

    output
    gv: numpy array 1D - auxiliary variable.

    References:

    Clark, D. A., S. J. Saul and D. W. Emerson (1986),
    Magnetic and gravity anomalies of a triaxial ellipsoid.
    '''

    assert a > b > c, 'a must be greater than b and b must be greater \
        than c'

    assert (a > 0) and (b > 0) and (c > 0), 'a, b and c must \
        be all positive'

    assert v in ['x', 'y', 'z'], "v must be 'x', 'y' or 'z'"

    if v is 'x':
        E, F = _E_F_field(a, b, c, kappa, phi)
        aux1 = 2./((a*a - b*b)*np.sqrt(a*a - c*c))
        gv = aux1*(F - E)

    if v is 'y':
        E, F = _E_F_field(a, b, c, kappa, phi)
        aux1 = 2*np.sqrt(a*a - c*c)/((a*a - b*b)*(b*b - c*c))
        aux2 = (b*b - c*c)/(a*a - c*c)
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)
        aux3 = ((kappa*kappa)*sinphi*cosphi) /\
            np.sqrt(1. - (kappa*sinphi*kappa*sinphi))
        gv = aux1*(E - aux2*F - aux3)

    if v is 'z':
        E, F = _E_F_field(a, b, c, kappa, phi)
        aux1 = 2./((b*b - c*c)*np.sqrt(a*a - c*c))
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)
        aux2 = (sinphi*np.sqrt(1. - (kappa*sinphi*kappa*sinphi)))/cosphi
        gv = aux1*(aux2 - E)

    return gv


def _gv_tejedor(a, b, c, kappa, phi, lamb, v='x'):
    '''
    Diagonal terms of the depolarization tensor defined outside the
    ellipsoidal body. These terms depend on the Legendre's normal
    elliptic integrals of first and second kinds (Tejedor, 1995).

    input
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    c: float - semi-axis c (in meters).
    kappa: numpy array 1D - parameter of the elliptic integral.
    phi: numpy array 1D - amplitude of the elliptic integral.
    lambda: float - parameter lambda defining the surface of the triaxial
        ellipsoid.
    v: string - defines the coordinate with respect to which the
        variable gv will be calculated. It must be 'x', 'y' or 'z'.

    output
    gv: numpy array 1D - auxiliary variable.

    References:

    Tejedor, M., Rubio, H., Elbaile, L., and Iglesias, R.: External
    fields created by uniformly magnetized ellipsoids and spheroids,
    IEEE transactions on magnetics, 31, 830-836, 1995.
    '''

    assert a > b > c, 'a must be greater than b and b must be greater \
        than c'

    assert (a > 0) and (b > 0) and (c > 0), 'a, b and c must \
        be all positive'

    assert v in ['x', 'y', 'z'], "v must be 'x', 'y' or 'z'"

    if v is 'x':
        E, F = _E_F_field(a, b, c, kappa, phi)
        aux1 = 2/(np.sqrt(a*a - c*c)*(a*a - b*b))
        gv = aux1*(F - E)

    if v is 'y':
        E, F = _E_F_field(a, b, c, kappa, phi)
        aux1 = (2*np.sqrt(a*a - c*c))/((a*a - b*b)*(b*b - c*c))
        aux2 = -2/(np.sqrt(a*a - c*c)*(a*a - b*b))
        aux3 = (-2/(b*b - c*c))*np.sqrt((c*c + lamb) /
                                        ((a*a + lamb)*(b*b + lamb)))
        gv = aux1*E + aux2*F + aux3

    if v is 'z':
        E, F = _E_F_field(a, b, c, kappa, phi)
        aux1 = 2/(np.sqrt(a*a - c*c)*(c*c - b*b))
        aux2 = (2/(b*b - c*c))*np.sqrt((b*b + lamb) /
                                       ((a*a + lamb)*(c*c + lamb)))
        gv = aux1*E + aux2

    return gv


def _nuv(xp, yp, zp, ellipsoid, u='x', v='x'):
    r"""
    The uv element of the depolarization tensor evaluated
    outside the body.

    The coordinate system of the input parameters is x -> semi-axis a,
    y -> semi-axis b and z -> semi-axis c.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the element will be calculated.
    * ellipsoid : element of :class:`fatiando.mesher.EllipsoidTriaxial`
        The ellipsoid. The ellipsoid must have the physical property
        ``'k'`` and/or ``'remanence'``. If the ellipsoids is ``None`` or
        without ``'k'`` and ``'remanence'``, it is ignored.
    * u, v : strings
        Define the element to be calculated.

    Returns:

    * nuv : array
        The uv component of the depolarization tensor evaluated outsice
        the body, in the ellipsoid system.

    """

    assert xp.size == yp.size == zp.size, \
        'xp, yp and zp must have the same size'

    assert u in ['x', 'y', 'z'], 'u must be x, y or z'
    assert v in ['x', 'y', 'z'], 'v must be x, y or z'

    alpha, gamma, delta = structural_angles(ellipsoid.strike, ellipsoid.dip,
                                            ellipsoid.rake)
    matrix = V(alpha, gamma, delta)

    x1, x2, x3 = x1x2x3(xp, yp, zp, ellipsoid.x, ellipsoid.y, ellipsoid.z,
                        matrix)
    lamb = _lamb(x1, x2, x3, ellipsoid.a, ellipsoid.b, ellipsoid.c)
    denominator = _dlamb_aux(x1, x2, x3,
                             ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb)
    dlamb = _dlamb(x1, x2, x3, ellipsoid.a, ellipsoid.b, ellipsoid.c,
                   lamb, denominator, deriv=u)
    h = _hv(ellipsoid.a, ellipsoid.b, ellipsoid.c, lamb, v=v)
    aux = -0.5*ellipsoid.a*ellipsoid.b*ellipsoid.c
    if v == 'x':
        res = 8*np.pi*dlamb*h*x1
    if v == 'y':
        res = 8*np.pi*dlamb*h*x2
    if v == 'z':
        res = 8*np.pi*dlamb*h*x3
    if v == u:
        kappa, phi = _E_F_field_args(ellipsoid.a, ellipsoid.b, ellipsoid.c,
                                     lamb)
        g = _gv_tejedor(ellipsoid.a, ellipsoid.b, ellipsoid.c,
                        kappa, phi, lamb, v=v)
        res += 4*np.pi*g

    return aux*res
