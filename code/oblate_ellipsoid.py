r"""
The potential fields of a homogeneous oblate ellipsoid.
"""
from __future__ import division

import numpy as np

from ..constants import SI2MGAL, G, CM, T2NT, SI2EOTVOS
from .. import utils
from .._our_duecredit import due, Doi


due.cite(Doi("XXXXXXXXXXXXXXXXX"),
         description='Forward modeling formula for oblate ellipsoids.',
         path='fatiando.gravmag.oblate_ellipsoid')


def tf(xp, yp, zp, ellipsoids, F, inc, dec, pmag=None):
    r"""
    The total-field anomaly produced by oblate ellipsoids
    (Emerson et al., 1985).

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
    * ellipsoids : list of :class:`fatiando.mesher.EllipsoidOblate`
        The ellipsoids. Ellipsoids must have the physical property
        ``'k'`` and/or ``'remanence'``. Ellipsoids that are ``None`` or
        without ``'k'`` and ``'remanence'`` will be ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of the resultant magnetization of the ellipsoids. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * tf : array
        The total-field anomaly

    References:

    Emerson, D. W., Clark, D., and Saul, S.: Magnetic exploration models
    incorporating remanence, demagnetization and anisotropy: HP 41C handheld
    computer algorithms, Exploration Geophysics, 16, 1-122, 1985.

    """
    fx, fy, fz = utils.dircos(inc, dec)
    Bx = bx(xp, yp, zp, ellipsoids, F, inc, dec, pmag)
    By = by(xp, yp, zp, ellipsoids, F, inc, dec, pmag)
    Bz = bz(xp, yp, zp, ellipsoids, F, inc, dec, pmag)

    return fx*Bx + fy*By + fz*Bz


def bx(xp, yp, zp, ellipsoids, F, inc, dec, pmag=None):
    r"""
    The x component of the magnetoc induction produced by oblate
    ellipsoids (Emerson et al., 1985).

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoids : list of :class:`fatiando.mesher.EllipsoidOblate`
        The ellipsoids. Ellipsoids must have the physical property
        ``'k'`` and/or ``'remanence'``. Ellipsoids that are ``None`` or
        without ``'k'`` and ``'remanence'`` will be ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of the resultant magnetization of the ellipsoids. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * bx: array
        The x component of the magnetic induction

    References:

    Emerson, D. W., Clark, D., and Saul, S.: Magnetic exploration models
    incorporating remanence, demagnetization and anisotropy: HP 41C handheld
    computer algorithms, Exploration Geophysics, 16, 1-122, 1985.

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
        b1 = _bx(xp, yp, zp, ellipsoid, F, inc, dec, pmag)
        b2 = _by(xp, yp, zp, ellipsoid, F, inc, dec, pmag)
        b3 = _bz(xp, yp, zp, ellipsoid, F, inc, dec, pmag)
        res += matrix[0, 0]*b1 + matrix[0, 1]*b2 + matrix[0, 2]*b3

    return res


def by(xp, yp, zp, ellipsoids, F, inc, dec, pmag=None):
    r"""
    The y component of the magnetoc induction produced by oblate
    ellipsoids (Emerson et al., 1985).

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoids : list of :class:`fatiando.mesher.EllipsoidOblate`
        The ellipsoids. Ellipsoids must have the physical property
        ``'k'`` and/or ``'remanence'``. Ellipsoids that are ``None`` or
        without ``'k'`` and ``'remanence'`` will be ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of the resultant magnetization of the ellipsoids. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * by: array
        The y component of the magnetic induction

    References:

    Emerson, D. W., Clark, D., and Saul, S.: Magnetic exploration models
    incorporating remanence, demagnetization and anisotropy: HP 41C handheld
    computer algorithms, Exploration Geophysics, 16, 1-122, 1985.

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
        b1 = _bx(xp, yp, zp, ellipsoid, F, inc, dec, pmag)
        b2 = _by(xp, yp, zp, ellipsoid, F, inc, dec, pmag)
        b3 = _bz(xp, yp, zp, ellipsoid, F, inc, dec, pmag)
        res += matrix[1, 0]*b1 + matrix[1, 1]*b2 + matrix[1, 2]*b3

    return res


def bz(xp, yp, zp, ellipsoids, F, inc, dec, pmag=None):
    r"""
    The z component of the magnetoc induction produced by oblate
    ellipsoids (Emerson et al., 1985).

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoids : list of :class:`fatiando.mesher.EllipsoidOblate`
        The ellipsoids. Ellipsoids must have the physical property
        ``'k'`` and/or ``'remanence'``. Ellipsoids that are ``None`` or
        without ``'k'`` and ``'remanence'`` will be ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of the resultant magnetization of the ellipsoids. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * bz: array
        The z component of the magnetic induction

    References:

    Emerson, D. W., Clark, D., and Saul, S.: Magnetic exploration models
    incorporating remanence, demagnetization and anisotropy: HP 41C handheld
    computer algorithms, Exploration Geophysics, 16, 1-122, 1985.

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
        b1 = _bx(xp, yp, zp, ellipsoid, F, inc, dec, pmag)
        b2 = _by(xp, yp, zp, ellipsoid, F, inc, dec, pmag)
        b3 = _bz(xp, yp, zp, ellipsoid, F, inc, dec, pmag)
        res += matrix[2, 0]*b1 + matrix[2, 1]*b2 + matrix[2, 2]*b3
    return res


def _bx(xp, yp, zp, ellipsoid, F, inc, dec, pmag=None):
    r"""
    The x component of the magnetoc induction produced by oblate
    ellipsoids in the ellipsoid system (Emerson et al., 1985).

    The coordinate system of the input parameters is x -> semi-axis a,
    y -> semi-axis b and z -> semi-axis c.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoid : element of :class:`fatiando.mesher.EllipsoidOblate`
        The ellipsoid. The ellipsoid must have the physical property
        ``'k'`` and/or ``'remanence'``. If the ellipsoids is ``None`` or
        without ``'k'`` and ``'remanence'``, it is ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of the resultant magnetization of the ellipsoid. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * bx : array
        The x component of the magnetic induction in the ellipsoid system.

    References:

    Emerson, D. W., Clark, D., and Saul, S.: Magnetic exploration models
    incorporating remanence, demagnetization and anisotropy: HP 41C handheld
    computer algorithms, Exploration Geophysics, 16, 1-122, 1985.

    """

    alpha, gamma, delta = structural_angles(ellipsoid.strike,
                                            ellipsoid.dip, ellipsoid.rake)
    matrix = V(alpha, gamma, delta)

    if pmag is None:
        n11, n22 = demag_factors(ellipsoid.a, ellipsoid.b)
        k1, k2, k3, alpha_susc, gamma_susc, delta_susc = ellipsoid.props['k']
        suscep = K(k1, k2, k3, alpha_susc, gamma_susc, delta_susc)
        if 'remanence' in ellipsoid.props:
            Hr, incr, decr = ellipsoid.props['remanence']
        else:
            Hr, incr, decr = 0, 0, 0
        mx, my, mz = magnetization(n11, n22, suscep, F, inc, dec,
                                   Hr, incr, decr, matrix)
    else:
        mx, my, mz = pmag

    x1, x2, x3 = x1x2x3(xp, yp, zp, ellipsoid.x, ellipsoid.y, ellipsoid.z,
                        matrix)
    lamb = _lamb(x1, x2, x3, ellipsoid.a, ellipsoid.b)
    denominator = _dlamb_aux(x1, x2, x3,
                             ellipsoid.a, ellipsoid.b, lamb)
    dlamb = _dlamb(x1, x2, x3, ellipsoid.a, ellipsoid.b,
                   lamb, denominator, deriv='x')
    h1 = _hv(ellipsoid.a, ellipsoid.b, lamb, v='x')
    h2 = _hv(ellipsoid.a, ellipsoid.b, lamb, v='y')
    h3 = _hv(ellipsoid.a, ellipsoid.b, lamb, v='z')
    g = _gv(ellipsoid.a, ellipsoid.b, lamb, v='x')
    res = dlamb*(h1*x1*mx + h2*x2*my + h3*x3*mz) + g*mx
    res *= -CM*T2NT*2*np.pi*ellipsoid.a*ellipsoid.b*ellipsoid.b
    return res


def _by(xp, yp, zp, ellipsoid, F, inc, dec, pmag=None):
    r"""
    The y component of the magnetoc induction produced by oblate
    ellipsoids in the ellipsoid system (Emerson et al., 1985).

    The coordinate system of the input parameters is x -> semi-axis a,
    y -> semi-axis b and z -> semi-axis c.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoid : element of :class:`fatiando.mesher.EllipsoidOblate`
        The ellipsoid. The ellipsoid must have the physical property
        ``'k'`` and/or ``'remanence'``. If the ellipsoids is ``None`` or
        without ``'k'`` and ``'remanence'``, it is ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of the resultant magnetization of the ellipsoid. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * by : array
        The y component of the magnetic induction in the ellipsoid system.

    References:

    Emerson, D. W., Clark, D., and Saul, S.: Magnetic exploration models
    incorporating remanence, demagnetization and anisotropy: HP 41C handheld
    computer algorithms, Exploration Geophysics, 16, 1-122, 1985.

    """

    alpha, gamma, delta = structural_angles(ellipsoid.strike,
                                            ellipsoid.dip, ellipsoid.rake)
    matrix = V(alpha, gamma, delta)

    if pmag is None:
        n11, n22 = demag_factors(ellipsoid.a, ellipsoid.b)
        k1, k2, k3, alpha_susc, gamma_susc, delta_susc = ellipsoid.props['k']
        suscep = K(k1, k2, k3, alpha_susc, gamma_susc, delta_susc)
        if 'remanence' in ellipsoid.props:
            Hr, incr, decr = ellipsoid.props['remanence']
        else:
            Hr, incr, decr = 0, 0, 0
        mx, my, mz = magnetization(n11, n22, suscep, F, inc, dec,
                                   Hr, incr, decr, matrix)
    else:
        mx, my, mz = pmag

    x1, x2, x3 = x1x2x3(xp, yp, zp, ellipsoid.x, ellipsoid.y, ellipsoid.z,
                        matrix)
    lamb = _lamb(x1, x2, x3, ellipsoid.a, ellipsoid.b)
    denominator = _dlamb_aux(x1, x2, x3,
                             ellipsoid.a, ellipsoid.b, lamb)
    dlamb = _dlamb(x1, x2, x3, ellipsoid.a, ellipsoid.b,
                   lamb, denominator, deriv='y')
    h1 = _hv(ellipsoid.a, ellipsoid.b, lamb, v='x')
    h2 = _hv(ellipsoid.a, ellipsoid.b, lamb, v='y')
    h3 = _hv(ellipsoid.a, ellipsoid.b, lamb, v='z')
    g = _gv(ellipsoid.a, ellipsoid.b, lamb, v='y')
    res = dlamb*(h1*x1*mx + h2*x2*my + h3*x3*mz) + g*my
    res *= -CM*T2NT*2*np.pi*ellipsoid.a*ellipsoid.b*ellipsoid.b
    return res


def _bz(xp, yp, zp, ellipsoid, F, inc, dec, pmag=None):
    r"""
    The z component of the magnetoc induction produced by oblate
    ellipsoids in the ellipsoid system (Emerson et al., 1985).

    The coordinate system of the input parameters is x -> semi-axis a,
    y -> semi-axis b and z -> semi-axis c.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoid : element of :class:`fatiando.mesher.EllipsoidOblate`
        The ellipsoid. The ellipsoid must have the physical property
        ``'k'`` and/or ``'remanence'``. If the ellipsoids is ``None`` or
        without ``'k'`` and ``'remanence'``, it is ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of the resultant magnetization of the ellipsoid. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * bz : array
        The z component of the magnetic induction in the ellipsoid system.

    References:

    Emerson, D. W., Clark, D., and Saul, S.: Magnetic exploration models
    incorporating remanence, demagnetization and anisotropy: HP 41C handheld
    computer algorithms, Exploration Geophysics, 16, 1-122, 1985.

    """

    alpha, gamma, delta = structural_angles(ellipsoid.strike,
                                            ellipsoid.dip, ellipsoid.rake)
    matrix = V(alpha, gamma, delta)

    if pmag is None:
        n11, n22 = demag_factors(ellipsoid.a, ellipsoid.b)
        k1, k2, k3, alpha_susc, gamma_susc, delta_susc = ellipsoid.props['k']
        suscep = K(k1, k2, k3, alpha_susc, gamma_susc, delta_susc)
        if 'remanence' in ellipsoid.props:
            Hr, incr, decr = ellipsoid.props['remanence']
        else:
            Hr, incr, decr = 0, 0, 0
        mx, my, mz = magnetization(n11, n22, suscep, F, inc, dec,
                                   Hr, incr, decr, matrix)
    else:
        mx, my, mz = pmag

    x1, x2, x3 = x1x2x3(xp, yp, zp, ellipsoid.x, ellipsoid.y, ellipsoid.z,
                        matrix)
    lamb = _lamb(x1, x2, x3, ellipsoid.a, ellipsoid.b)
    denominator = _dlamb_aux(x1, x2, x3,
                             ellipsoid.a, ellipsoid.b, lamb)
    dlamb = _dlamb(x1, x2, x3, ellipsoid.a, ellipsoid.b,
                   lamb, denominator, deriv='z')
    h1 = _hv(ellipsoid.a, ellipsoid.b, lamb, v='x')
    h2 = _hv(ellipsoid.a, ellipsoid.b, lamb, v='y')
    h3 = _hv(ellipsoid.a, ellipsoid.b, lamb, v='z')
    g = _gv(ellipsoid.a, ellipsoid.b, lamb, v='z')
    res = dlamb*(h1*x1*mx + h2*x2*my + h3*x3*mz) + g*mz
    res *= -CM*T2NT*2*np.pi*ellipsoid.a*ellipsoid.b*ellipsoid.b
    return res


def structural_angles(strike, dip, rake):
    '''
    Calculates the orientation angles alpha, gamma
    and delta (Emerson et al., 1985)
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

    Emerson, D. W., Clark, D., and Saul, S.: Magnetic exploration models
    incorporating remanence, demagnetization and anisotropy: HP 41C
    handheld computer algorithms, Exploration Geophysics, 16, 1-122, 1985.

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
    for a oblate model. The columns of this matrix
    are defined according to the unit vectors v1, v2
    and v3.

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
    alpha: float - orientation angle (in radians) defining the
    susceptibility tensor.
    gamma: float - orientation angle (in radians) defining the
    susceptibility tensor.
    delta: float - orientation angle (in radians) defining the
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


def _lamb(x1, x2, x3, a, b):
    '''
    Calculates the parameter lambda.

    input
    x: numpy array 1D - x coordinates in the ellipsoid system (in meters).
    y: numpy array 1D - y coordinates in the ellipsoid system (in meters).
    z: numpy array 1D - z coordinates in the ellipsoid system (in meters).
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).

    output
    lamb: numpy array 1D - parameter lambda for each point in the
        ellipsoid system.
    '''

    assert b > a, 'b must be greater than a'

    assert (a > 0) and (b > 0), 'a and b must be positive'

    # auxiliary variables
    p1 = a*a + b*b - x1*x1 - x2*x2 - x3*x3
    p0 = a*a*b*b - b*b*x1*x1 - a*a*(x2*x2 + x3*x3)

    delta = np.sqrt(p1*p1 - 4*p0)

    lamb = (-p1 + delta)/2.

    return lamb


def _quadratic_coeffs(x, y, z, a, b):
    '''
    Calculates the coefficients of the quadratic equation defining
    a oblate ellipsoid.

    input
    x: numpy array 1D - x coordinates in the ellipsoid system (in meters).
    y: numpy array 1D - y coordinates in the ellipsoid system (in meters).
    z: numpy array 1D - z coordinates in the ellipsoid system (in meters).
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).

    output
    p1: numpy array 1D - coefficient multiplying the linear term.
    p0: numpy array 1D - coefficient multiplying the constant.
    '''

    assert b > a, 'b must be greater than a'

    assert (a > 0) and (b > 0), 'a, b and c must be all positive'

    # auxiliary variables
    p1 = a*a + b*b - x*x - y*y - z*z
    p0 = (a*b*a*b) - (b*x*b*x) - a*a*(y*y + z*z)

    return p1, p0


def _dlamb(x, y, z, a, b, lamb, denominator, deriv='x'):
    '''
    Calculates the spatial derivative of the parameter lambda
    with respect to the coordinates x, y or z in the ellipsoid system.

    input
    x: numpy array 1D - x coordinates in the ellipsoid system (in meters).
    y: numpy array 1D - y coordinates in the ellipsoid system (in meters).
    z: numpy array 1D - z coordinates in the ellipsoid system (in meters).
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    lambda: numpy array 1D - parameter lambda defining the surface of the
        oblate ellipsoid.
    denominator: numpy array 1D - denominator of the equation used to
        calculate the spatial derivative of lambda.
    deriv: string - defines the coordinate with respect to which the
        derivative will be calculated. It must be 'x', 'y' or 'z'.

    output
    dlamb_dv: numpy array 1D - derivative of lambda with respect to the
        coordinate v = x, y, z in the ellipsoid system.
    '''

    assert b > a, 'b must be greater than a'

    assert (a > 0) and (b > 0), 'a and b must be all positive'

    assert deriv in ['x', 'y', 'z'], 'deriv must represent a coordinate \
        x, y or z'

    assert denominator.size == lamb.size == x.size == y.size == z.size, \
        'x, y, z, lamb and denominator must have the same size'

    if deriv is 'x':
        dlamb_dv = (2*x/(a*a + lamb))/denominator

    if deriv is 'y':
        dlamb_dv = (2*y/(b*b + lamb))/denominator

    if deriv is 'z':
        dlamb_dv = (2*z/(b*b + lamb))/denominator

    return dlamb_dv


def _dlamb_aux(x, y, z, a, b, lamb):
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
    lambda: float - parameter lambda defining the surface of the oblate
        ellipsoid.

    output
    aux: numpy array 1D - denominator of the expression used to calculate
        the spatial derivative of lambda.
    '''

    assert b > a, 'b must be greater than a'

    assert (a > 0) and (b > 0), 'a and b must be all positive'

    aux1 = x/(a*a + lamb)
    aux2 = y/(b*b + lamb)
    aux3 = z/(b*b + lamb)
    aux = aux1*aux1 + aux2*aux2 + aux3*aux3

    return aux


def demag_factors(a, b):
    '''
    Calculates the demagnetizing factors n11 and n22.

    input
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).

    output
    n11: float - demagnetizing factor along the semi-axis a (in SI).
    n22: float - demagnetizing factor along the semi-axis b (in SI).
    '''

    assert b > a, 'b must be greater than a'

    assert (a > 0) and (b > 0), 'a and b must be all positive'

    m = a/b

    aux1 = 1 - m*m
    aux2 = np.sqrt(aux1)

    n11 = (1/aux1)*(1 - (m*np.arccos(m))/aux2)
    n22 = 0.5*(1 - n11)

    return n11, n22


def magnetization(n11, n22, suscep, F, inc, dec, RM, incrm, decrm, matrix):
    '''
    Calculates the resultant magnetization corrected from
    demagnetizing in the ellipsoid system.

    input
    n11: float - demagnetizing factor along the semi-axis a (in SI).
    n22: float - demagnetizing factor along the semi-axis b (in SI).
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

    output
    m: numpy array 1D - resultant magnetization (in A/m) in the
        ellipsoid system.
    '''

    assert np.allclose(np.dot(matrix.T, matrix), np.identity(3)), \
        'matrix must be a valid coordinate transformation matrix'

    assert n11 >= n22, 'n11 must be greater than n22'

    assert (n11 >= 0) and (n22 >= 0), 'n11 and n22 must \
be all positive or zero (for neglecting the self-demagnetization)'

    assert np.allclose(suscep.T, suscep), 'the susceptibility is a \
symmetrical tensor'

    N_tilde = np.diag([n11, n22, n22])
    suscep_tilde = np.dot(matrix.T, np.dot(suscep, matrix))
    H0_tilde = np.dot(matrix.T, utils.ang2vec(F/(4*np.pi*100), inc, dec))
    RM_tilde = np.dot(matrix.T, utils.ang2vec(RM, incrm, decrm))

    # resultant magnetization in the ellipsoid system
    M_tilde = np.linalg.solve(np.identity(3) - np.dot(suscep_tilde, N_tilde),
                              np.dot(suscep_tilde, H0_tilde) + RM_tilde)

    return M_tilde


def _hv(a, b, lamb, v='x'):
    '''
    Calculates an auxiliary variable used to calculate the
    depolarization tensor outside the ellipsoidal body.

    input
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    lambda: float - parameter lambda defining the surface of the oblate
        ellipsoid.
    v: string - defines the coordinate with respect to which the
        variable hv will be calculated. It must be 'x', 'y' or 'z'.

    output
    hv: numpy array 1D - auxiliary variable.
    '''

    assert b > a, 'b must be greater than a'

    assert (a > 0) and (b > 0), 'a, b and c must be all positive'

    assert v in ['x', 'y', 'z'], "v must be 'x', 'y' or 'z'"

    aux1 = a*a + lamb
    aux2 = b*b + lamb

    if v is 'x':
        hv = -1./(np.sqrt(aux1*aux1*aux1)*aux2)

    if v is 'y' or 'z':
        hv = -1./(np.sqrt(aux1)*aux2*aux2)

    return hv


def _gv(a, b, lamb, v='x'):
    '''
    Diagonal terms of the depolarization tensor defined outside the
    ellipsoidal body.

    input
    a: float - semi-axis a (in meters).
    b: float - semi-axis b (in meters).
    lambda: float - parameter lambda defining the surface of the oblate
        ellipsoid.
    v: string - defines the coordinate with respect to which the
        variable gv will be calculated. It must be 'x', 'y' or 'z'.

    output
    gv: numpy array 1D - auxiliary variable.
    '''

    assert b > a, 'b must be greater than a'

    assert (a > 0) and (b > 0), 'a and b must be all positive'

    assert v in ['x', 'y', 'z'], "v must be 'x', 'y' or 'z'"

    atan = np.arctan(np.sqrt((b*b-a*a)/(a*a+lamb)))
    aux1 = 1./np.sqrt((b*b - a*a)*(b*b - a*a)*(b*b - a*a))

    if v is 'x':
        aux2 = np.sqrt((b*b - a*a)/(a*a + lamb))
        gv = 2*aux1*(aux2 - atan)

    if v is 'y' or 'z':
        aux2 = np.sqrt((b*b - a*a)*(a*a + lamb))/(b*b + lamb)
        gv = aux1*(atan - aux2)

    return gv
