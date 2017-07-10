from __future__ import division, absolute_import
import numpy as np
from copy import deepcopy

from fatiando import utils, gridder
from mesher import OblateEllipsoid
import oblate_ellipsoid
from numpy.testing import assert_almost_equal
from pytest import raises

# Local-geomagnetic field
F = 30000
inc = 2
dec = -27

gm = 1000  # geometrical factor
area = [-5.*gm, 5.*gm, -5.*gm, 5.*gm]
x, y, z = gridder.scatter(area, 300, z=0.)
axis_ref = gm  # reference semi-axis

# Oblate ellipsoids used for testing
model = [OblateEllipsoid(x=-3*gm, y=-3*gm, z=3*axis_ref,
                         small_axis=0.6*axis_ref,
                         large_axis=axis_ref,
                         strike=78, dip=92, rake=135,
                         props={'principal susceptibilities': [0.7, 0.7,
                                                               0.7],
                                'susceptibility angles': [90., 47., 13.]}),
         OblateEllipsoid(x=-gm, y=-gm, z=2.4*axis_ref,
                         small_axis=0.3*axis_ref,
                         large_axis=1.1*axis_ref,
                         strike=4, dip=10, rake=5,
                         props={'principal susceptibilities': [0.2, 0.15,
                                                               0.05],
                                'susceptibility angles': [180, 19, -8.],
                                'remanent magnetization': [3, -6, 35]}),
         OblateEllipsoid(x=3*gm, y=3*gm, z=4*axis_ref,
                         small_axis=0.6*axis_ref,
                         large_axis=1.5*axis_ref,
                         strike=-58, dip=87, rake=49,
                         props={'remanent magnetization': [4.7, 39, 0]})]


def test_oblate_ellipsoid_force_prop():
    "Test the oblate_ellipsoid code with an imposed physical property"

    # forced physical property
    pmag = utils.ang2vec(5, 43, -8)

    # magnetic field produced by the ellipsoids
    # with the forced physical property
    bx = oblate_ellipsoid.bx(x, y, z, model,
                             F, inc, dec, demag=False, pmag=pmag)
    by = oblate_ellipsoid.by(x, y, z, model,
                             F, inc, dec, demag=False, pmag=pmag)
    bz = oblate_ellipsoid.bz(x, y, z, model,
                             F, inc, dec, demag=False, pmag=pmag)
    tf = oblate_ellipsoid.tf(x, y, z, model,
                             F, inc, dec, demag=False, pmag=pmag)

    # constant factor
    f = 3.71768

    # magnetic field produced by the ellipsoids
    # with the forced physical property multiplied by the constant factor
    bx2 = oblate_ellipsoid.bx(x, y, z, model,
                              F, inc, dec, demag=False, pmag=f*pmag)
    by2 = oblate_ellipsoid.by(x, y, z, model,
                              F, inc, dec, demag=False, pmag=f*pmag)
    bz2 = oblate_ellipsoid.bz(x, y, z, model,
                              F, inc, dec, demag=False, pmag=f*pmag)
    tf2 = oblate_ellipsoid.tf(x, y, z, model,
                              F, inc, dec, demag=False, pmag=f*pmag)

    # the fields must be proportional
    assert_almost_equal(bx2, f*bx, decimal=10)
    assert_almost_equal(by2, f*by, decimal=10)
    assert_almost_equal(bz2, f*bz, decimal=10)
    assert_almost_equal(tf2, f*tf, decimal=10)

    # pmag not None requires demag not True
    raises(AssertionError, oblate_ellipsoid.bx, x, y, z, model,
           F, inc, dec, demag=True, pmag=pmag)
    raises(AssertionError, oblate_ellipsoid.by, x, y, z, model,
           F, inc, dec, demag=True, pmag=pmag)
    raises(AssertionError, oblate_ellipsoid.bz, x, y, z, model,
           F, inc, dec, demag=True, pmag=pmag)
    raises(AssertionError, oblate_ellipsoid.tf, x, y, z, model,
           F, inc, dec, demag=True, pmag=pmag)


def test_oblate_ellipsoid_ignore_none():
    "Oblate ellipsoid ignores model elements that are None"

    # forced physical property
    pmag = utils.ang2vec(7, -52, 13)

    # copy of the original model
    model_none = deepcopy(model)

    # force an element of the copy to be None
    model_none[1] = None

    # magnetic field produced by the original model
    # without the removed element
    bx = oblate_ellipsoid.bx(x, y, z, [model[0], model[2]],
                             F, inc, dec, demag=False, pmag=pmag)
    by = oblate_ellipsoid.by(x, y, z, [model[0], model[2]],
                             F, inc, dec, demag=False, pmag=pmag)
    bz = oblate_ellipsoid.bz(x, y, z, [model[0], model[2]],
                             F, inc, dec, demag=False, pmag=pmag)
    tf = oblate_ellipsoid.tf(x, y, z, [model[0], model[2]],
                             F, inc, dec, demag=False, pmag=pmag)

    # magnetic field produced by the copy
    bx2 = oblate_ellipsoid.bx(x, y, z, model_none,
                              F, inc, dec, demag=False, pmag=pmag)
    by2 = oblate_ellipsoid.by(x, y, z, model_none,
                              F, inc, dec, demag=False, pmag=pmag)
    bz2 = oblate_ellipsoid.bz(x, y, z, model_none,
                              F, inc, dec, demag=False, pmag=pmag)
    tf2 = oblate_ellipsoid.tf(x, y, z, model_none,
                              F, inc, dec, demag=False, pmag=pmag)

    assert_almost_equal(bx2, bx, decimal=15)
    assert_almost_equal(by2, by, decimal=15)
    assert_almost_equal(bz2, bz, decimal=15)
    assert_almost_equal(tf2, tf, decimal=15)


def test_oblate_ellipsoid_missing_prop():
    "Self-demagnetization requires specific properties"

    # demag=True requires specific properties
    raises(AssertionError, oblate_ellipsoid._bx, x, y, z, model[2],
           F, inc, dec, demag=True)
    raises(AssertionError, oblate_ellipsoid._by, x, y, z, model[2],
           F, inc, dec, demag=True)
    raises(AssertionError, oblate_ellipsoid._bz, x, y, z, model[2],
           F, inc, dec, demag=True)


def test_oblate_ellipsoid_susceptibility_tensor_missing_prop():
    "Susceptibility tensor requires specific properties"

    suscep1 = model[0].susceptibility_tensor
    suscep2 = model[1].susceptibility_tensor
    suscep3 = model[2].susceptibility_tensor
    assert suscep1 is not None
    assert suscep2 is not None
    assert suscep3 is None


def test_oblate_ellipsoid_demag_factors_sum():
    "The summation of the demagnetizing factors must be equal to one"

    n11, n22 = oblate_ellipsoid.demag_factors(model[0])
    assert_almost_equal(n11+n22+n22, 1., decimal=15)

    n11, n22 = oblate_ellipsoid.demag_factors(model[1])
    assert_almost_equal(n11+n22+n22, 1., decimal=15)

    n11, n22 = oblate_ellipsoid.demag_factors(model[2])
    assert_almost_equal(n11+n22+n22, 1., decimal=15)


def test_oblate_ellipsoid_demag_factors_signal_order():
    "Demagnetizing factors must be all positive and ordered"

    n11, n22 = oblate_ellipsoid.demag_factors(model[0])
    assert (n11 > 0) and (n22 > 0)
    assert n11 > n22

    n11, n22 = oblate_ellipsoid.demag_factors(model[1])
    assert (n11 > 0) and (n22 > 0)
    assert n11 > n22

    n11, n22 = oblate_ellipsoid.demag_factors(model[2])
    assert (n11 > 0) and (n22 > 0)
    assert n11 > n22


def test_oblate_ellipsoid_self_demagnetization():
    "Self-demagnetization decreases the magnetization intensity"

    mag_with_demag = oblate_ellipsoid.magnetization(model[1],
                                                    F, inc, dec,
                                                    demag=True)

    mag_without_demag = oblate_ellipsoid.magnetization(model[1],
                                                       F, inc, dec,
                                                       demag=False)

    mag_with_demag_norm = np.linalg.norm(mag_with_demag, ord=2)
    mag_without_demag_norm = np.linalg.norm(mag_without_demag, ord=2)

    assert mag_with_demag_norm < mag_without_demag_norm


def test_oblate_ellipsoid_neglecting_self_demagnetization():
    "The error in magnetization by negleting self-demagnetization is bounded"

    # susceptibility tensor
    k1, k2, k3 = model[0].props['principal susceptibilities']
    strike, dip, rake = model[0].props['susceptibility angles']

    # demagnetizing factors
    n11, n22 = oblate_ellipsoid.demag_factors(model[0])

    # maximum relative error in the resulting magnetization
    max_error = k3*n11

    # magnetizations calculated with and without self-demagnetization
    mag_with_demag = oblate_ellipsoid.magnetization(model[0],
                                                    F, inc, dec,
                                                    demag=True)
    mag_without_demag = oblate_ellipsoid.magnetization(model[0],
                                                       F, inc, dec,
                                                       demag=False)

    # difference in magnetization
    mag_diff = mag_with_demag - mag_without_demag

    # computed norms
    mag_with_demag_norm = np.linalg.norm(mag_with_demag, ord=2)
    mag_diff_norm = np.linalg.norm(mag_diff, ord=2)

    # computed error
    computed_error = mag_diff_norm/mag_with_demag_norm

    assert computed_error <= max_error


def test_oblate_ellipsoid_depolarization_tensor():
    "The depolarization tensor must be symmetric"

    ellipsoid = model[1]
    x1, x2, x3 = oblate_ellipsoid.x1x2x3(x, y, z, ellipsoid)
    lamb = oblate_ellipsoid._lamb(x1, x2, x3, ellipsoid)
    denominator = oblate_ellipsoid._dlamb_aux(x1, x2, x3, ellipsoid, lamb)
    dlamb_dx = oblate_ellipsoid._dlamb(x1, x2, x3, ellipsoid, lamb,
                                       denominator, deriv='x')
    dlamb_dy = oblate_ellipsoid._dlamb(x1, x2, x3, ellipsoid, lamb,
                                       denominator, deriv='y')
    dlamb_dz = oblate_ellipsoid._dlamb(x1, x2, x3, ellipsoid, lamb,
                                       denominator, deriv='z')
    h1 = oblate_ellipsoid._hv(ellipsoid, lamb, v='x')
    h2 = oblate_ellipsoid._hv(ellipsoid, lamb, v='y')
    h3 = oblate_ellipsoid._hv(ellipsoid, lamb, v='z')
    g1 = oblate_ellipsoid._gv(ellipsoid, lamb, v='x')
    g2 = oblate_ellipsoid._gv(ellipsoid, lamb, v='y')
    g3 = oblate_ellipsoid._gv(ellipsoid, lamb, v='z')
    a = ellipsoid.large_axis
    b = ellipsoid.small_axis
    cte = -0.5*a*b*b

    # elements of the depolarization tensor without the ellipsoid
    nxx = cte*(dlamb_dx*h1*x1 + g1)
    nyy = cte*(dlamb_dy*h2*x2 + g2)
    nzz = cte*(dlamb_dz*h3*x3 + g3)
    nxy = cte*(dlamb_dx*h2*x2)
    nyx = cte*(dlamb_dy*h1*x1)
    nxz = cte*(dlamb_dx*h3*x3)
    nzx = cte*(dlamb_dz*h1*x1)
    nyz = cte*(dlamb_dy*h3*x3)
    nzy = cte*(dlamb_dz*h2*x2)
    trace = nxx+nyy+nzz

    # the trace must zero
    assert_almost_equal(trace, np.zeros_like(nxx), decimal=3)

    # the depolarization is symmetric
    assert_almost_equal(nxy, nyx, decimal=3)
    assert_almost_equal(nxz, nzx, decimal=3)
    assert_almost_equal(nyz, nzy, decimal=3)


def test_oblate_ellipsoid_isotropic_susceptibility():
    "Isostropic susceptibility must be proportional to identity"

    k1, k2, k3 = model[0].props['principal susceptibilities']
    strike, dip, rake = model[0].props['susceptibility angles']
    suscep = model[0].susceptibility_tensor
    assert np.allclose(suscep, k1*np.identity(3))


def test_confocal_oblate_ellipsoids():
    "Confocal bodies with properly scaled suscep produce the same field"
    # Reference ellipsoid
    a, b, = 400., 1000.  # semi-axes
    chi = 1.2  # reference susceptibility
    ellipsoid = OblateEllipsoid(0., 0., 1500., a, b, 45., 10., -30.,
                                {'principal susceptibilities': [chi,
                                                                chi,
                                                                chi],
                                 'susceptibility angles': [0., 0., 0.]})
    # Intensity of the local-geomagnetic field (in nT)
    B0 = 23500.
    # Direction parallel to the semi-axis a
    _, inc, dec = utils.vec2ang(ellipsoid.transf_matrix.T[0])
    # Magnetic moment of the reference ellipsoid
    volume = ellipsoid.volume
    mag = oblate_ellipsoid.magnetization(ellipsoid, B0,
                                         inc, dec, demag=True)
    moment = volume*mag
    # Confocal ellipsoid
    u = 2.0e6
    a_confocal = np.sqrt(a*a + u)
    b_confocal = np.sqrt(b*b + u)
    xc = ellipsoid.x
    yc = ellipsoid.y
    zc = ellipsoid.z
    strike = ellipsoid.strike
    dip = ellipsoid.dip
    rake = ellipsoid.rake
    confocal_ellipsoid = OblateEllipsoid(xc, yc, zc,
                                         a_confocal, b_confocal,
                                         strike, dip, rake,
                                         {'susceptibility angles':
                                          [0., 0., 0.]})
    n11, n22 = oblate_ellipsoid.demag_factors(confocal_ellipsoid)
    H0 = B0/(4*np.pi*100)
    volume_confocal = confocal_ellipsoid.volume
    # Equivalent susceptibility
    moment_norm = np.sqrt(np.sum(moment*moment))
    chi_confocal = moment_norm/(volume_confocal*H0 - n11*moment_norm)
    confocal_ellipsoid.addprop('principal susceptibilities',
                               [chi_confocal, chi_confocal, chi_confocal])
    # Magnetic moment of the confocal ellipsoid
    mag_confocal = oblate_ellipsoid.magnetization(confocal_ellipsoid, B0,
                                                  inc, dec, demag=True)
    moment_confocal = volume_confocal*mag_confocal
    # Total-field anomalies
    tf = oblate_ellipsoid.tf(x, y, z, [ellipsoid], B0, inc, dec)
    tf_confocal = oblate_ellipsoid.tf(x, y, z, [confocal_ellipsoid],
                                      B0, inc, dec)
    # Comparison between the moments and total-field anomalies
    assert_almost_equal(moment, moment_confocal, decimal=5)
    assert_almost_equal(tf, tf_confocal, decimal=12)
