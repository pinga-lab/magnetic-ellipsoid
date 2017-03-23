r"""
The potential fields of a homogeneous triaxial ellipsoid.
"""
from __future__ import division, absolute_import

import numpy as np
from scipy.special import ellipeinc, ellipkinc

from fatiando.constants import PERM_FREE_SPACE, T2NT
from fatiando import utils


def tf(xp, yp, zp, ellipsoids, F, inc, dec, demag=True, pmag=None):
    r"""
    The total-field anomaly produced by triaxial ellipsoids.

    .. math::

        \Delta T = |\mathbf{T}| - |\mathbf{F}|,

    where :math:`\mathbf{T}` is the measured field and :math:`\mathbf{F}` is
    the local-geomagnetic field.

    The anomaly of a homogeneous ellipsoid can be calculated as:

    .. math::

        \Delta T \approx \hat{\mathbf{F}}\cdot\mathbf{B}.

    where :math:`\mathbf{B}` is the magnetic induction produced by the
    ellipsoid.

    This code follows the approach presented by Clark et al. (1986).

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated.
    * ellipsoids : list of :class:`mesher.TriaxialEllipsoid`
        The ellipsoids. Ellipsoids must have the physical properties
        ``'principal susceptibilities'`` and ``'susceptibility angles'``
        as prerequisite to calculate the self-demagnetization.
        Ellipsoids that are ``None`` will be ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * demag : boolean
        If True, will include the self-demagnetization.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of calculating the magnetization of the ellipsoid. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * tf : array
        The total-field anomaly

    References:

    Clark, D., Saul, S., and Emerson, D.: Magnetic and gravity anomalies
    of a triaxial ellipsoid, Exploration Geophysics, 17, 189-200, 1986.

    """
    fx, fy, fz = utils.dircos(inc, dec)
    Bx = bx(xp, yp, zp, ellipsoids, F, inc, dec, demag, pmag)
    By = by(xp, yp, zp, ellipsoids, F, inc, dec, demag, pmag)
    Bz = bz(xp, yp, zp, ellipsoids, F, inc, dec, demag, pmag)

    return fx*Bx + fy*By + fz*Bz


def bx(xp, yp, zp, ellipsoids, F, inc, dec, demag=True, pmag=None):
    r"""
    The x component of the magnetic induction produced by triaxial
    ellipsoids.

    This code follows the approach presented by Clark et al. (1986).

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoids : list of :class:`mesher.TriaxialEllipsoid`
        The ellipsoids. Ellipsoids must have the physical properties
        ``'principal susceptibilities'`` and ``'susceptibility angles'``
        as prerequisite to calculate the self-demagnetization.
        Ellipsoids that are ``None`` will be ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * demag : boolean
        If True, will include the self-demagnetization.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of the resultant magnetization of the ellipsoids. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * bx: array
        The x component of the magnetic induction

    References:

    Clark, D., Saul, S., and Emerson, D.: Magnetic and gravity anomalies
    of a triaxial ellipsoid, Exploration Geophysics, 17, 189-200, 1986.

    """

    res = 0
    for ellipsoid in ellipsoids:
        if ellipsoid is None:
            continue
        b1 = _bx(xp, yp, zp, ellipsoid, F, inc, dec, demag, pmag)
        b2 = _by(xp, yp, zp, ellipsoid, F, inc, dec, demag, pmag)
        b3 = _bz(xp, yp, zp, ellipsoid, F, inc, dec, demag, pmag)
        res += ellipsoid.transf_matrix[0, 0]*b1 \
            + ellipsoid.transf_matrix[0, 1]*b2 \
            + ellipsoid.transf_matrix[0, 2]*b3

    return res


def by(xp, yp, zp, ellipsoids, F, inc, dec, demag=True, pmag=None):
    r"""
    The y component of the magnetic induction produced by triaxial
    ellipsoids.

    This code follows the approach presented by Clark et al. (1986).

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoids : list of :class:`mesher.TriaxialEllipsoid`
        The ellipsoids. Ellipsoids must have the physical properties
        ``'principal susceptibilities'`` and ``'susceptibility angles'``
        as prerequisite to calculate the self-demagnetization.
        Ellipsoids that are ``None`` will be ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * demag : boolean
        If True, will include the self-demagnetization.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of the resultant magnetization of the ellipsoids. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * by: array
        The y component of the magnetic induction

    References:

    Clark, D., Saul, S., and Emerson, D.: Magnetic and gravity anomalies
    of a triaxial ellipsoid, Exploration Geophysics, 17, 189-200, 1986.

    """

    res = 0
    for ellipsoid in ellipsoids:
        if ellipsoid is None:
            continue
        b1 = _bx(xp, yp, zp, ellipsoid, F, inc, dec, demag, pmag)
        b2 = _by(xp, yp, zp, ellipsoid, F, inc, dec, demag, pmag)
        b3 = _bz(xp, yp, zp, ellipsoid, F, inc, dec, demag, pmag)
        res += ellipsoid.transf_matrix[1, 0]*b1 \
            + ellipsoid.transf_matrix[1, 1]*b2 \
            + ellipsoid.transf_matrix[1, 2]*b3

    return res


def bz(xp, yp, zp, ellipsoids, F, inc, dec, demag=True, pmag=None):
    r"""
    The z component of the magnetic induction produced by triaxial
    ellipsoids.

    This code follows the approach presented by Clark et al. (1986).

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoids : list of :class:`mesher.TriaxialEllipsoid`
        The ellipsoids. Ellipsoids must have the physical properties
        ``'principal susceptibilities'`` and ``'susceptibility angles'``
        as prerequisite to calculate the self-demagnetization.
        Ellipsoids that are ``None`` will be ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * demag : boolean
        If True, will include the self-demagnetization.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of the resultant magnetization of the ellipsoids. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * bz: array
        The z component of the magnetic induction

    References:

    Clark, D., Saul, S., and Emerson, D.: Magnetic and gravity anomalies
    of a triaxial ellipsoid, Exploration Geophysics, 17, 189-200, 1986.

    """

    res = 0
    for ellipsoid in ellipsoids:
        if ellipsoid is None:
            continue
        b1 = _bx(xp, yp, zp, ellipsoid, F, inc, dec, demag, pmag)
        b2 = _by(xp, yp, zp, ellipsoid, F, inc, dec, demag, pmag)
        b3 = _bz(xp, yp, zp, ellipsoid, F, inc, dec, demag, pmag)
        res += ellipsoid.transf_matrix[2, 0]*b1 \
            + ellipsoid.transf_matrix[2, 1]*b2 \
            + ellipsoid.transf_matrix[2, 2]*b3

    return res


def _bx(xp, yp, zp, ellipsoid, F, inc, dec, demag=True, pmag=None):
    r"""
    The x component of the magnetic induction produced by triaxial
    ellipsoids in the ellipsoid system.

    This code follows the approach presented by Clark et al. (1986).

    The coordinate system of the input parameters is x -> semi-axis a,
    y -> semi-axis b and z -> semi-axis c.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoid : element of :class:`mesher.TriaxialEllipsoid`.
        The ellipsoid. It must have the physical properties
        ``'principal susceptibilities'`` and ``'susceptibility angles'``
        as prerequisite to calculate the self-demagnetization.
        If the ellipsoid is ``None``, it will be ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * demag : boolean
        If True, include the self-demagnetization.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of the resultant magnetization of the ellipsoid. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * bx : array
        The x component of the magnetic induction in the ellipsoid system.

    References:

    Clark, D., Saul, S., and Emerson, D.: Magnetic and gravity anomalies
    of a triaxial ellipsoid, Exploration Geophysics, 17, 189-200, 1986.

    """

    if pmag is None:
        mx, my, mz = magnetization(ellipsoid, F, inc, dec, demag)
    else:
        assert demag is not True, 'the use of a forced magnetization \
impedes the computation of self-demagnetization'
        mx, my, mz = pmag

    x1, x2, x3 = x1x2x3(xp, yp, zp, ellipsoid)
    lamb = _lamb(x1, x2, x3, ellipsoid)
    denominator = _dlamb_aux(x1, x2, x3, ellipsoid, lamb)
    dlamb = _dlamb(x1, x2, x3, ellipsoid, lamb, denominator, deriv='x')
    h1 = _hv(ellipsoid, lamb, v='x')
    h2 = _hv(ellipsoid, lamb, v='y')
    h3 = _hv(ellipsoid, lamb, v='z')
    kappa, phi = _E_F_field_args(ellipsoid, lamb)
    g = _gv_tejedor(ellipsoid, kappa, phi, lamb, v='x')

    res = dlamb*(h1*x1*mx + h2*x2*my + h3*x3*mz) + g*mx
    a = ellipsoid.large_axis
    b = ellipsoid.intermediate_axis
    c = ellipsoid.small_axis
    res *= -0.5*a*b*c

    res *= -PERM_FREE_SPACE*T2NT

    return res


def _by(xp, yp, zp, ellipsoid, F, inc, dec, demag=True, pmag=None):
    r"""
    The y component of the magnetic induction produced by triaxial
    ellipsoids in the ellipsoid system.

    This code follows the approach presented by Clark et al. (1986).

    The coordinate system of the input parameters is x -> semi-axis a,
    y -> semi-axis b and z -> semi-axis c.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoid : element of :class:`mesher.TriaxialEllipsoid`
        The ellipsoid. It must have the physical properties
        ``'principal susceptibilities'`` and ``'susceptibility angles'``
        as prerequisite to calculate the self-demagnetization.
        If the ellipsoid is ``None``, it will be ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * demag : boolean
        If True, include the self-demagnetization.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of the resultant magnetization of the ellipsoid. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * by : array
        The y component of the magnetic induction in the ellipsoid system.

    References:

    Clark, D., Saul, S., and Emerson, D.: Magnetic and gravity anomalies
    of a triaxial ellipsoid, Exploration Geophysics, 17, 189-200, 1986.

    """

    if pmag is None:
        mx, my, mz = magnetization(ellipsoid, F, inc, dec, demag)
    else:
        assert demag is not True, 'the use of a forced magnetization \
impedes the computation of self-demagnetization'
        mx, my, mz = pmag

    x1, x2, x3 = x1x2x3(xp, yp, zp, ellipsoid)
    lamb = _lamb(x1, x2, x3, ellipsoid)
    denominator = _dlamb_aux(x1, x2, x3, ellipsoid, lamb)
    dlamb = _dlamb(x1, x2, x3, ellipsoid, lamb, denominator, deriv='y')
    h1 = _hv(ellipsoid, lamb, v='x')
    h2 = _hv(ellipsoid, lamb, v='y')
    h3 = _hv(ellipsoid, lamb, v='z')
    kappa, phi = _E_F_field_args(ellipsoid, lamb)
    g = _gv_tejedor(ellipsoid, kappa, phi, lamb, v='y')

    res = dlamb*(h1*x1*mx + h2*x2*my + h3*x3*mz) + g*my
    a = ellipsoid.large_axis
    b = ellipsoid.intermediate_axis
    c = ellipsoid.small_axis
    res *= -0.5*a*b*c

    res *= -PERM_FREE_SPACE*T2NT

    return res


def _bz(xp, yp, zp, ellipsoid, F, inc, dec, demag=True, pmag=None):
    r"""
    The z component of the magnetic induction produced by triaxial
    ellipsoids in the ellipsoid system.

    This code follows the approach presented by Clark et al. (1986).

    The coordinate system of the input parameters is x -> semi-axis a,
    y -> semi-axis b and z -> semi-axis c.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoid : element of :class:`mesher.TriaxialEllipsoid`
        The ellipsoid. It must have the physical properties
        ``'principal susceptibilities'`` and ``'susceptibility angles'``
        as prerequisite to calculate the self-demagnetization.
        If the ellipsoid is ``None``, it will be ignored.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * demag : boolean
        If True, will include the self-demagnetization.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead
        of the resultant magnetization of the ellipsoid. Use this, e.g.,
        for sensitivity matrix building.

    Returns:

    * bz : array
        The z component of the magnetic induction in the ellipsoid system.

    References:

    Clark, D., Saul, S., and Emerson, D.: Magnetic and gravity anomalies
    of a triaxial ellipsoid, Exploration Geophysics, 17, 189-200, 1986.

    """

    if pmag is None:
        mx, my, mz = magnetization(ellipsoid, F, inc, dec, demag)
    else:
        assert demag is not True, 'the use of a forced magnetization \
impedes the computation of self-demagnetization'
        mx, my, mz = pmag

    x1, x2, x3 = x1x2x3(xp, yp, zp, ellipsoid)
    lamb = _lamb(x1, x2, x3, ellipsoid)
    denominator = _dlamb_aux(x1, x2, x3, ellipsoid, lamb)
    dlamb = _dlamb(x1, x2, x3, ellipsoid, lamb, denominator, deriv='z')
    h1 = _hv(ellipsoid, lamb, v='x')
    h2 = _hv(ellipsoid, lamb, v='y')
    h3 = _hv(ellipsoid, lamb, v='z')
    kappa, phi = _E_F_field_args(ellipsoid, lamb)
    g = _gv_tejedor(ellipsoid, kappa, phi, lamb, v='z')

    res = dlamb*(h1*x1*mx + h2*x2*my + h3*x3*mz) + g*mz
    a = ellipsoid.large_axis
    b = ellipsoid.intermediate_axis
    c = ellipsoid.small_axis
    res *= -0.5*a*b*c

    res *= -PERM_FREE_SPACE*T2NT

    return res


def x1x2x3(xp, yp, zp, ellipsoid):
    '''
    Calculates the x, y and z coordinates referred to the
    ellipsoid coordinate system.

    Parameters:

    * xp, yp, zp: numpy arrays 1D
        x, y and z coordinates of points referred to the main
        system (in meters).
    * ellipsoid : element of :class:`mesher.TriaxialEllipsoid`.

    Returns:

    * x1, x2, x3: numpy arrays 1D
        x, y and z coordinates of points referred to the ellipsoid
        system (in meters).
    '''

    assert xp.size == yp.size == zp.size, \
        'xp, yp and zp must have the same size'

    x1 = ellipsoid.transf_matrix[0, 0]*(xp - ellipsoid.x) + \
        ellipsoid.transf_matrix[1, 0]*(yp - ellipsoid.y) + \
        ellipsoid.transf_matrix[2, 0]*(zp - ellipsoid.z)
    x2 = ellipsoid.transf_matrix[0, 1]*(xp - ellipsoid.x) + \
        ellipsoid.transf_matrix[1, 1]*(yp - ellipsoid.y) + \
        ellipsoid.transf_matrix[2, 1]*(zp - ellipsoid.z)
    x3 = ellipsoid.transf_matrix[0, 2]*(xp - ellipsoid.x) + \
        ellipsoid.transf_matrix[1, 2]*(yp - ellipsoid.y) + \
        ellipsoid.transf_matrix[2, 2]*(zp - ellipsoid.z)

    return x1, x2, x3


def _lamb(x1, x2, x3, ellipsoid):
    '''
    Calculates the parameter lambda for a triaxial ellispoid.

    The parameter lambda is defined as the largest root of
    the cubic equation defining the surface of the triaxial
    ellipsoid.

    Parameters:

    * x1, x2, x3: numpy arrays 1D
        x, y and z coordinates of points referred to the ellipsoid
        system (in meters).
    * ellipsoid : element of :class:`mesher.TriaxialEllipsoid`.

    Returns:

    * lamb: numpy array 1D
        Parameter lambda for each point in the ellipsoid system.
    '''

    a = ellipsoid.large_axis
    b = ellipsoid.intermediate_axis
    c = ellipsoid.small_axis

    # auxiliary variables (http://mathworld.wolfram.com/CubicFormula.html)
    x1x1 = x1*x1
    x2x2 = x2*x2
    x3x3 = x3*x3
    p2 = a*a + b*b + c*c - x1x1 - x2x2 - x3x3
    p1 = (b*c*b*c) + (a*c*a*c) + (a*b*a*b) - (b*b + c*c)*x1x1 \
        - (a*a + c*c)*x2x2 - (a*a + b*b)*x3x3
    p0 = (a*b*c*a*b*c) - (b*c*b*c*x1x1) - (a*c*a*c*x2x2) - (a*b*a*b*x3x3)
    Q = (3.*p1 - p2*p2)/9.
    R = (9.*p1*p2 - 27.*p0 - 2.*p2*p2*p2)/54.

    p3 = R/np.sqrt(-(Q*Q*Q))

    assert np.alltrue(p3 <= 1.), 'arccos argument greater than 1'

    assert np.alltrue(Q*Q*Q + R*R < 0), 'the polynomial discriminant \
must be negative'

    theta = np.arccos(p3)

    lamb = 2.*np.sqrt(-Q)*np.cos(theta/3.) - p2/3.

    return lamb


def _dlamb(x1, x2, x3, ellipsoid, lamb, denominator, deriv='x'):
    '''
    Calculates the spatial derivative of the parameter lambda
    with respect to the coordinates x, y or z in the ellipsoid system.

    Parameters:

    * x1, x2, x3: numpy arrays 1D
        x, y and z coordinates of points referred to the ellipsoid
        system (in meters).
    * ellipsoid : element of :class:`mesher.TriaxialEllipsoid`.
    * lamb: numpy array 1D
        Parameter lambda for each point in the ellipsoid system.
    * denominator: numpy array 1D
        Denominator of the expression used to calculate the spatial
        derivative of the parameter lambda.
    * deriv: string
        Defines the coordinate with respect to which the
        derivative will be calculated. It must be 'x', 'y' or 'z'.

    Returns:

    * dlamb_dv: numpy array 1D
        Derivative of lambda with respect to the coordinate
        v = x, y, z in the ellipsoid system.
    '''

    assert deriv in ['x', 'y', 'z'], 'deriv must represent a coordinate \
        x, y or z'

    assert denominator.size == lamb.size == x1.size == x2.size == x3.size, \
        'x1, x2, x3, lamb and denominator must have the same size'

    a = ellipsoid.large_axis
    b = ellipsoid.intermediate_axis
    c = ellipsoid.small_axis

    if deriv is 'x':
        dlamb_dv = (2*x1/(a*a + lamb))/denominator

    if deriv is 'y':
        dlamb_dv = (2*x2/(b*b + lamb))/denominator

    if deriv is 'z':
        dlamb_dv = (2*x3/(c*c + lamb))/denominator

    return dlamb_dv


def _dlamb_aux(x1, x2, x3, ellipsoid, lamb):
    '''
    Calculates an auxiliary variable used to calculate the spatial
    derivatives of the parameter lambda with respect to the
    coordinates x, y and z in the ellipsoid system.

    Parameters:

    * x1, x2, x3: numpy arrays 1D
        x, y and z coordinates of points referred to the ellipsoid
        system (in meters).
    * ellipsoid : element of :class:`mesher.TriaxialEllipsoid`.
    * lamb: numpy array 1D
        Parameter lambda for each point in the ellipsoid system.

    Returns:

    * aux: numpy array 1D
        Denominator of the expression used to calculate the spatial
        derivative of the parameter lambda.
    '''

    a = ellipsoid.large_axis
    b = ellipsoid.intermediate_axis
    c = ellipsoid.small_axis

    aux1 = x1/(a*a + lamb)
    aux2 = x2/(b*b + lamb)
    aux3 = x3/(c*c + lamb)
    aux = aux1*aux1 + aux2*aux2 + aux3*aux3

    return aux


def _E_F_demag(ellipsoid):
    '''
    Calculates the Legendre's normal elliptic integrals of first
    and second kinds which are used to calculate the demagnetizing
    factors.

    Parameters:

    * ellipsoid : element of :class:`mesher.TriaxialEllipsoid`.

    Returns:

    F, E : floats
        Legendre's normal elliptic integrals of first and second kinds,
        respectively.
    '''

    a = ellipsoid.large_axis
    b = ellipsoid.intermediate_axis
    c = ellipsoid.small_axis

    kappa = (a*a-b*b)/(a*a-c*c)
    phi = np.arccos(c/a)

    # E = ellipeinc(phi, kappa*kappa)
    # F = ellipkinc(phi, kappa*kappa)
    E = ellipeinc(phi, kappa)
    F = ellipkinc(phi, kappa)

    return E, F


def demag_factors(ellipsoid):
    '''
    Calculates the demagnetizing factors n11, n22 and n33.

    Parameters:

    * ellipsoid : element of :class:`mesher.TriaxialEllipsoid`.

    Returns:

    * n11, n22, n33: floats
        Demagnetizing factors (in SI) along, respectively, the
        large, intermediate and small axes of the triaxial ellipsoid.
    '''

    E, F = _E_F_demag(ellipsoid)

    a = ellipsoid.large_axis
    b = ellipsoid.intermediate_axis
    c = ellipsoid.small_axis

    aux1 = (a*b*c)/np.sqrt((a*a - c*c))
    n11 = (aux1/(a*a - b*b))*(F - E)
    n22 = -n11 + (aux1/(b*b - c*c))*E - (c*c)/(b*b - c*c)
    n33 = -(aux1/(b*b - c*c))*E + (b*b)/(b*b - c*c)

    return n11, n22, n33


def magnetization(ellipsoid, F, inc, dec, demag):
    '''
    Calculates the resultant magnetization corrected from
    demagnetizing in the main system.

    Parameters:

    * ellipsoid: element of :class:`mesher.TriaxialEllipsoid`.
    * F, inc, dec : floats
       The intensity (in nT), inclination and declination (in degrees) of
       the local-geomagnetic field.
    * demag : boolean
        If True, will include the self-demagnetization.

    Returns:

    * resultant_mag: numpy array 1D
        Resultant magnetization (in A/m) in the main system.
    '''

    # Remanent magnetization
    if 'remanent magnetization' in ellipsoid.props:
        intensity = ellipsoid.props['remanent magnetization'][0]
        inclination = ellipsoid.props['remanent magnetization'][1]
        declination = ellipsoid.props['remanent magnetization'][2]
        remanent_mag = utils.ang2vec(intensity, inclination, declination)
    else:
        remanent_mag = np.zeros(3)

    suscep = ellipsoid.susceptibility_tensor

    # Induced magnetization
    if suscep is not None:
        geomag_field = utils.ang2vec(F/(4*np.pi*100), inc, dec)
        induced_mag = np.dot(suscep, geomag_field)
    else:
        induced_mag = np.zeros(3)

    # Self-demagnetization
    if demag is True:

        assert suscep is not None, 'self-demagnetization requires a \
susceptibility tensor'

        n11, n22, n33 = demag_factors(ellipsoid)
        coord_transf_matrix = ellipsoid.transf_matrix
        suscep_tilde = np.dot(np.dot(coord_transf_matrix.T, suscep),
                              coord_transf_matrix)
        aux = np.linalg.inv(np.identity(3) + np.dot(suscep_tilde,
                                                    np.diag([n11, n22, n33])))
        Lambda = np.dot(np.dot(coord_transf_matrix, aux),
                        coord_transf_matrix.T)

        # resultant magnetization in the main system
        resultant_mag = np.dot(Lambda, induced_mag + remanent_mag)

    else:

        assert (suscep is not None) or ('remanent magnetization'
                                        in ellipsoid.props), 'neglecting \
            self-demagnetization requires a susceptibility tensor or a rem\
            anent magnetization'

        # resultant magnetization in the main system
        resultant_mag = induced_mag + remanent_mag

    return resultant_mag


def _E_F_field(ellipsoid, kappa, phi):
    '''
    Calculates the Legendre's normal elliptic integrals of first
    and second kinds which are used to calculate the potential
    fields outside the triaxial ellipsoid.

    Parameters:

    * ellipsoid : element of :class:`mesher.TriaxialEllipsoid`.
    * lamb: numpy array 1D
        Parameter lambda for each point in the ellipsoid system.
    * kappa: numpy array 1D
        Squared modulus of the elliptic integral.
    * phi: numpy array 1D
        Amplitude of the elliptic integral.

    Returns:

    F, E: numpy arrays 1D
        Legendre's normal elliptic integrals of first and second kinds.
    '''

    E = ellipeinc(phi, kappa)
    F = ellipkinc(phi, kappa)

    return E, F


def _E_F_field_args(ellipsoid, lamb):
    '''
    Calculates the arguments of the elliptic integrals defining
    the elements of the depolarization tensor without the body.

    Parameters:

    * ellipsoid : element of :class:`mesher.TriaxialEllipsoid`.
    * lamb: numpy array 1D
        Parameter lambda for each point in the ellipsoid system.

    Returns:

    * kappa: numpy array 1D
        Squared modulus of the elliptic integral.
    * phi: numpy array 1D
        Amplitude of the elliptic integral.
    '''

    a = ellipsoid.large_axis
    b = ellipsoid.intermediate_axis
    c = ellipsoid.small_axis

    kappa = (a*a - b*b)/(a*a - c*c)
    phi = np.arcsin(np.sqrt((a*a - c*c)/(a*a + lamb)))

    return kappa, phi


def _hv(ellipsoid, lamb, v='x'):
    '''
    Calculates an auxiliary variable used to calculate the
    depolarization tensor outside the ellipsoidal body.

    Parameters:

    * ellipsoid : element of :class:`mesher.TriaxialEllipsoid`.
    * lamb: numpy array 1D
        Parameter lambda for each point in the ellipsoid system.
    * v: string
        Defines the coordinate with respect to which the
        variable hv will be calculated. It must be 'x', 'y' or 'z'.

    Returns:

    * hv: numpy array 1D
        Auxiliary variable.
    '''

    assert v in ['x', 'y', 'z'], "v must be 'x', 'y' or 'z'"

    a = ellipsoid.large_axis
    b = ellipsoid.intermediate_axis
    c = ellipsoid.small_axis

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


def _gv(ellipsoid, kappa, phi, v='x'):
    '''
    Diagonal term of the depolarization tensor defined outside the
    ellipsoidal body. This term depends on the Legendre's normal
    elliptic integrals of first and second kinds (Clark, 1986).

    Parameters:

    * ellipsoid : element of :class:`mesher.TriaxialEllipsoid`.
    * lamb: numpy array 1D
        Parameter lambda for each point in the ellipsoid system.
    * kappa: numpy array 1D
        Squared modulus of the elliptic integral.
    * phi: numpy array 1D
        Amplitude of the elliptic integral.
    * v: string
        Defines the coordinate with respect to which the
        variable gv will be calculated. It must be 'x', 'y' or 'z'.

    Returns:

    * gv: numpy array 1D
        Diagonal term of the depolarization tensor calculated for
        each lambda.

    References:

    Clark, D., Saul, S., and Emerson, D.: Magnetic and gravity anomalies
    of a triaxial ellipsoid, Exploration Geophysics, 17, 189-200, 1986.
    '''

    assert v in ['x', 'y', 'z'], "v must be 'x', 'y' or 'z'"

    a = ellipsoid.large_axis
    b = ellipsoid.intermediate_axis
    c = ellipsoid.small_axis

    E, F = _E_F_field(ellipsoid, kappa, phi)

    if v is 'x':
        aux1 = 2./((a*a - b*b)*np.sqrt(a*a - c*c))
        gv = aux1*(F - E)

    if v is 'y':
        aux1 = 2*np.sqrt(a*a - c*c)/((a*a - b*b)*(b*b - c*c))
        aux2 = (b*b - c*c)/(a*a - c*c)
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)
        aux3 = (kappa*sinphi*cosphi) /\
            np.sqrt(1. - (kappa*sinphi*sinphi))
        gv = aux1*(E - aux2*F - aux3)

    if v is 'z':
        aux1 = 2./((b*b - c*c)*np.sqrt(a*a - c*c))
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)
        aux2 = (sinphi*np.sqrt(1. - (kappa*sinphi*sinphi)))/cosphi
        gv = aux1*(aux2 - E)

    return gv


def _gv_tejedor(ellipsoid, kappa, phi, lamb, v='x'):
    '''
    Diagonal term of the depolarization tensor defined outside the
    ellipsoidal body. This term depends on the Legendre's normal
    elliptic integrals of first and second kinds (Tejedor, 1995).

    Parameters:

    * ellipsoid : element of :class:`mesher.TriaxialEllipsoid`.
    * lamb: numpy array 1D
        Parameter lambda for each point in the ellipsoid system.
    * kappa: numpy array 1D
        Squared modulus of the elliptic integral.
    * phi: numpy array 1D
        Amplitude of the elliptic integral.
    * v: string
        Defines the coordinate with respect to which the
        variable gv will be calculated. It must be 'x', 'y' or 'z'.

    Returns:

    * gv: numpy array 1D
        Diagonal term of the depolarization tensor calculated for
        each lambda.

    References:

    Tejedor, M., Rubio, H., Elbaile, L., and Iglesias, R.: External
    fields created by uniformly magnetized ellipsoids and spheroids,
    IEEE transactions on magnetics, 31, 830-836, 1995.
    '''

    assert v in ['x', 'y', 'z'], "v must be 'x', 'y' or 'z'"

    a = ellipsoid.large_axis
    b = ellipsoid.intermediate_axis
    c = ellipsoid.small_axis

    E, F = _E_F_field(ellipsoid, kappa, phi)

    if v is 'x':
        aux1 = 2/(np.sqrt(a*a - c*c)*(a*a - b*b))
        gv = aux1*(F - E)

    if v is 'y':
        aux1 = (2*np.sqrt(a*a - c*c))/((a*a - b*b)*(b*b - c*c))
        aux2 = -2/(np.sqrt(a*a - c*c)*(a*a - b*b))
        aux3 = (-2/(b*b - c*c))*np.sqrt((c*c + lamb) /
                                        ((a*a + lamb)*(b*b + lamb)))
        gv = aux1*E + aux2*F + aux3

    if v is 'z':
        aux1 = 2/(np.sqrt(a*a - c*c)*(c*c - b*b))
        aux2 = (2/(b*b - c*c))*np.sqrt((b*b + lamb) /
                                       ((a*a + lamb)*(c*c + lamb)))
        gv = aux1*E + aux2

    return gv
