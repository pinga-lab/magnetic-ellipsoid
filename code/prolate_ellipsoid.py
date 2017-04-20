r"""
The potential fields of a homogeneous prolate ellipsoid.
"""
from __future__ import division, absolute_import

import numpy as np

from fatiando.constants import CM, T2NT
from fatiando import utils


def tf(xp, yp, zp, ellipsoids, F, inc, dec, demag=True, pmag=None):
    r"""
    The total-field anomaly produced by prolate ellipsoids.

    .. math::

        \Delta T = |\mathbf{T}| - |\mathbf{F}|,

    where :math:`\mathbf{T}` is the measured field and :math:`\mathbf{F}` is
    the local-geomagnetic field.

    The anomaly of a homogeneous ellipsoid can be calculated as:

    .. math::

        \Delta T \approx \hat{\mathbf{F}}\cdot\mathbf{B}.

    where :math:`\mathbf{B}` is the magnetic induction produced by the
    ellipsoid.

    This code follows the approach presented by Emerson et al. (1985).

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated.
    * ellipsoids : list of :class:`mesher.ProlateEllipsoid`
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

    Emerson, D. W., Clark, D., and Saul, S.: Magnetic exploration models
    incorporating remanence, demagnetization and anisotropy: HP 41C handheld
    computer algorithms, Exploration Geophysics, 16, 1-122, 1985.

    """
    fx, fy, fz = utils.dircos(inc, dec)
    Bx = bx(xp, yp, zp, ellipsoids, F, inc, dec, demag, pmag)
    By = by(xp, yp, zp, ellipsoids, F, inc, dec, demag, pmag)
    Bz = bz(xp, yp, zp, ellipsoids, F, inc, dec, demag, pmag)

    return fx*Bx + fy*By + fz*Bz


def bx(xp, yp, zp, ellipsoids, F, inc, dec, demag=True, pmag=None):
    r"""
    The x component of the magnetic induction produced by prolate
    ellipsoids.

    This code follows the approach presented by Emerson et al. (1985).

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoids : list of :class:`mesher.ProlateEllipsoid`
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

    Emerson, D. W., Clark, D., and Saul, S.: Magnetic exploration models
    incorporating remanence, demagnetization and anisotropy: HP 41C handheld
    computer algorithms, Exploration Geophysics, 16, 1-122, 1985.

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
    The y component of the magnetic induction produced by prolate
    ellipsoids.

    This code follows the approach presented by Emerson et al. (1985).

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoids : list of :class:`mesher.ProlateEllipsoid`
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

    Emerson, D. W., Clark, D., and Saul, S.: Magnetic exploration models
    incorporating remanence, demagnetization and anisotropy: HP 41C handheld
    computer algorithms, Exploration Geophysics, 16, 1-122, 1985.

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
    The z component of the magnetic induction produced by prolate
    ellipsoids.

    This code follows the approach presented by Emerson et al. (1985).

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoids : list of :class:`mesher.ProlateEllipsoid`
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

    Emerson, D. W., Clark, D., and Saul, S.: Magnetic exploration models
    incorporating remanence, demagnetization and anisotropy: HP 41C handheld
    computer algorithms, Exploration Geophysics, 16, 1-122, 1985.

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
    The x component of the magnetic induction produced by prolate
    ellipsoids in the ellipsoid system.

    This code follows the approach presented by Emerson et al. (1985).

    The coordinate system of the input parameters is x -> semi-axis a,
    y -> semi-axis b and z -> semi-axis c.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoid : element of :class:`mesher.ProlateEllipsoid`.
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

    Emerson, D. W., Clark, D., and Saul, S.: Magnetic exploration models
    incorporating remanence, demagnetization and anisotropy: HP 41C handheld
    computer algorithms, Exploration Geophysics, 16, 1-122, 1985.

    """

    if pmag is None:
        mx, my, mz = magnetization(ellipsoid, F, inc, dec, demag)
    else:
        assert demag is not True, 'the use of a forced magnetization \
impedes the computation of self-demagnetization'
        mx, my, mz = pmag

    # Transform the magnetization to the local coordinate system
    V = ellipsoid.transf_matrix
    mx_local = V[0, 0]*mx + V[1, 0]*my + V[2, 0]*mz
    my_local = V[0, 1]*mx + V[1, 1]*my + V[2, 1]*mz
    mz_local = V[0, 2]*mx + V[1, 2]*my + V[2, 2]*mz

    x1, x2, x3 = x1x2x3(xp, yp, zp, ellipsoid)
    lamb = _lamb(x1, x2, x3, ellipsoid)
    denominator = _dlamb_aux(x1, x2, x3, ellipsoid, lamb)
    dlamb = _dlamb(x1, x2, x3, ellipsoid, lamb, denominator, deriv='x')
    h1 = _hv(ellipsoid, lamb, v='x')
    h2 = _hv(ellipsoid, lamb, v='y')
    h3 = _hv(ellipsoid, lamb, v='z')
    g = _gv(ellipsoid, lamb, v='x')

    res = dlamb*(h1*x1*mx_local + h2*x2*my_local + h3*x3*mz_local)
    res += g*mx_local

    a = ellipsoid.large_axis
    b = ellipsoid.small_axis
    volume = 4*np.pi*a*b*b/3

    res *= -1.5*volume*CM*T2NT

    return res


def _by(xp, yp, zp, ellipsoid, F, inc, dec, demag=True, pmag=None):
    r"""
    The y component of the magnetic induction produced by prolate
    ellipsoids in the ellipsoid system.

    This code follows the approach presented by Emerson et al. (1985).

    The coordinate system of the input parameters is x -> semi-axis a,
    y -> semi-axis b and z -> semi-axis c.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoid : element of :class:`mesher.ProlateEllipsoid`
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

    Emerson, D. W., Clark, D., and Saul, S.: Magnetic exploration models
    incorporating remanence, demagnetization and anisotropy: HP 41C handheld
    computer algorithms, Exploration Geophysics, 16, 1-122, 1985.

    """

    if pmag is None:
        mx, my, mz = magnetization(ellipsoid, F, inc, dec, demag)
    else:
        assert demag is not True, 'the use of a forced magnetization \
impedes the computation of self-demagnetization'
        mx, my, mz = pmag

    # Transform the magnetization to the local coordinate system
    V = ellipsoid.transf_matrix
    mx_local = V[0, 0]*mx + V[1, 0]*my + V[2, 0]*mz
    my_local = V[0, 1]*mx + V[1, 1]*my + V[2, 1]*mz
    mz_local = V[0, 2]*mx + V[1, 2]*my + V[2, 2]*mz

    x1, x2, x3 = x1x2x3(xp, yp, zp, ellipsoid)
    lamb = _lamb(x1, x2, x3, ellipsoid)
    denominator = _dlamb_aux(x1, x2, x3, ellipsoid, lamb)
    dlamb = _dlamb(x1, x2, x3, ellipsoid, lamb, denominator, deriv='y')
    h1 = _hv(ellipsoid, lamb, v='x')
    h2 = _hv(ellipsoid, lamb, v='y')
    h3 = _hv(ellipsoid, lamb, v='z')
    g = _gv(ellipsoid, lamb, v='y')

    res = dlamb*(h1*x1*mx_local + h2*x2*my_local + h3*x3*mz_local)
    res += g*my_local

    a = ellipsoid.large_axis
    b = ellipsoid.small_axis
    volume = 4*np.pi*a*b*b/3

    res *= -1.5*volume*CM*T2NT

    return res


def _bz(xp, yp, zp, ellipsoid, F, inc, dec, demag=True, pmag=None):
    r"""
    The z component of the magnetic induction produced by prolate
    ellipsoids in the ellipsoid system.

    This code follows the approach presented by Emerson et al. (1985).

    The coordinate system of the input parameters is x -> semi-axis a,
    y -> semi-axis b and z -> semi-axis c.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * ellipsoid : element of :class:`mesher.ProlateEllipsoid`
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

    Emerson, D. W., Clark, D., and Saul, S.: Magnetic exploration models
    incorporating remanence, demagnetization and anisotropy: HP 41C handheld
    computer algorithms, Exploration Geophysics, 16, 1-122, 1985.

    """

    if pmag is None:
        mx, my, mz = magnetization(ellipsoid, F, inc, dec, demag)
    else:
        assert demag is not True, 'the use of a forced magnetization \
impedes the computation of self-demagnetization'
        mx, my, mz = pmag

    # Transform the magnetization to the local coordinate system
    V = ellipsoid.transf_matrix
    mx_local = V[0, 0]*mx + V[1, 0]*my + V[2, 0]*mz
    my_local = V[0, 1]*mx + V[1, 1]*my + V[2, 1]*mz
    mz_local = V[0, 2]*mx + V[1, 2]*my + V[2, 2]*mz

    x1, x2, x3 = x1x2x3(xp, yp, zp, ellipsoid)
    lamb = _lamb(x1, x2, x3, ellipsoid)
    denominator = _dlamb_aux(x1, x2, x3, ellipsoid, lamb)
    dlamb = _dlamb(x1, x2, x3, ellipsoid, lamb, denominator, deriv='z')
    h1 = _hv(ellipsoid, lamb, v='x')
    h2 = _hv(ellipsoid, lamb, v='y')
    h3 = _hv(ellipsoid, lamb, v='z')
    g = _gv(ellipsoid, lamb, v='z')

    res = dlamb*(h1*x1*mx_local + h2*x2*my_local + h3*x3*mz_local)
    res += g*mz_local

    a = ellipsoid.large_axis
    b = ellipsoid.small_axis
    volume = 4*np.pi*a*b*b/3

    res *= -1.5*volume*CM*T2NT

    return res


def x1x2x3(xp, yp, zp, ellipsoid):
    '''
    Calculates the x, y and z coordinates referred to the
    ellipsoid coordinate system.

    Parameters:

    * xp, yp, zp: numpy arrays 1D
        x, y and z coordinates of points referred to the main
        system (in meters).
    * ellipsoid : element of :class:`mesher.ProlateEllipsoid`.

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
    Calculates the parameter lambda for a prolate ellipsoid.

    The parameter lambda is defined as the largest root of
    the quadratic equation defining the surface of the prolate
    ellipsoid.

    Parameters:

    * x1, x2, x3: numpy arrays 1D
        x, y and z coordinates of points referred to the ellipsoid
        system (in meters).
    * ellipsoid : element of :class:`mesher.ProlateEllipsoid`.

    Returns:

    * lamb: numpy array 1D
        Parameter lambda for each point in the ellipsoid system.
    '''

    a = ellipsoid.large_axis
    b = ellipsoid.small_axis

    # auxiliary variables (http://mathworld.wolfram.com/QuadraticFormula.html)
    p1 = a*a + b*b - x1*x1 - x2*x2 - x3*x3
    p0 = a*a*b*b - b*b*x1*x1 - a*a*(x2*x2 + x3*x3)

    delta = np.sqrt(p1*p1 - 4*p0)

    lamb = (-p1 + delta)/2.

    return lamb


def _dlamb(x1, x2, x3, ellipsoid, lamb, denominator, deriv='x'):
    '''
    Calculates the spatial derivative of the parameter lambda
    with respect to the coordinates x, y or z in the ellipsoid system.

    Parameters:

    * x1, x2, x3: numpy arrays 1D
        x, y and z coordinates of points referred to the ellipsoid
        system (in meters).
    * ellipsoid : element of :class:`mesher.ProlateEllipsoid`.
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
    b = ellipsoid.small_axis

    if deriv is 'x':
        dlamb_dv = (2*x1/(a*a + lamb))/denominator

    if deriv is 'y':
        dlamb_dv = (2*x2/(b*b + lamb))/denominator

    if deriv is 'z':
        dlamb_dv = (2*x3/(b*b + lamb))/denominator

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
    * ellipsoid : element of :class:`mesher.ProlateEllipsoid`.
    * lamb: numpy array 1D
        Parameter lambda for each point in the ellipsoid system.

    Returns:

    * aux: numpy array 1D
        Denominator of the expression used to calculate the spatial
        derivative of the parameter lambda.
    '''

    a = ellipsoid.large_axis
    b = ellipsoid.small_axis

    aux1 = x1/(a*a + lamb)
    aux2 = x2/(b*b + lamb)
    aux3 = x3/(b*b + lamb)
    aux = aux1*aux1 + aux2*aux2 + aux3*aux3

    return aux


def demag_factors(ellipsoid):
    '''
    Calculates the demagnetizing factors n11 and n22.

    Parameters:

    * ellipsoid : element of :class:`mesher.ProlateEllipsoid`.

    Returns:

    * n11, n22: floats
        Demagnetizing factors (in SI) along, respectively, the
        large and small axes of the prolate ellipsoid.
    '''

    a = ellipsoid.large_axis
    b = ellipsoid.small_axis

    m = a/b

    aux1 = m*m - 1
    aux2 = np.sqrt(aux1)

    n11 = (1/aux1)*((m/aux2)*np.log(m + aux2) - 1)
    n22 = 0.5*(1 - n11)

    return n11, n22


def magnetization(ellipsoid, F, inc, dec, demag):
    '''
    Calculates the resultant magnetization corrected from
    demagnetizing in the main system.

    Parameters:

    * ellipsoid: element of :class:`mesher.ProlateEllipsoid`.
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

        n11, n22 = demag_factors(ellipsoid)
        coord_transf_matrix = ellipsoid.transf_matrix
        suscep_tilde = np.dot(np.dot(coord_transf_matrix.T, suscep),
                              coord_transf_matrix)
        aux = np.linalg.inv(np.identity(3) + np.dot(suscep_tilde,
                                                    np.diag([n11, n22, n22])))
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


def _hv(ellipsoid, lamb, v='x'):
    '''
    Calculates an auxiliary variable used to calculate the
    depolarization tensor outside the ellipsoidal body.

    Parameters:

    * ellipsoid : element of :class:`mesher.ProlateEllipsoid`.
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
    b = ellipsoid.small_axis

    aux1 = a*a + lamb
    aux2 = b*b + lamb

    if v is 'x':
        hv = -1./(np.sqrt(aux1*aux1*aux1)*aux2)

    if v is 'y' or 'z':
        hv = -1./(np.sqrt(aux1)*aux2*aux2)

    return hv


def _gv(ellipsoid, lamb, v='x'):
    '''
    Diagonal term of the depolarization tensor defined outside the
    ellipsoidal body.

    Parameters:

    * ellipsoid : element of :class:`mesher.ProlateEllipsoid`.
    * lamb: numpy array 1D
        Parameter lambda for each point in the ellipsoid system.
    * v: string
        Defines the coordinate with respect to which the
        variable gv will be calculated. It must be 'x', 'y' or 'z'.

    Returns:

    * gv: numpy array 1D
        Diagonal term of the depolarization tensor calculated for
        each lambda.
    '''

    assert v in ['x', 'y', 'z'], "v must be 'x', 'y' or 'z'"

    a = ellipsoid.large_axis
    b = ellipsoid.small_axis

    log = np.log((np.sqrt(a*a-b*b)+np.sqrt(a*a+lamb))/np.sqrt(b*b+lamb))
    aux1 = 1./np.sqrt((a*a - b*b)*(a*a - b*b)*(a*a - b*b))

    if v is 'x':
        aux2 = np.sqrt((a*a - b*b)/(a*a + lamb))
        gv = 2*aux1*(log - aux2)

    if v is 'y' or 'z':
        aux2 = np.sqrt((a*a - b*b)*(a*a + lamb))/(b*b + lamb)
        gv = aux1*(aux2 - log)

    return gv
