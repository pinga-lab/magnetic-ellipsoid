import numpy as np
from numpy.random import rand

from ... import utils, gridder
from ...mesher import OblateEllipsoid
from .. import oblate_ellipsoid

F = 20000 + 40000*rand()
inc = -90 + 180*rand()
dec = 180*rand()
sinc = -90 + 180*rand()
sdec = 180*rand()

gm = 1e4*rand()  # geometrical factor
area = [-5.*gm, 5.*gm, -5.*gm, 5.*gm]
x, y, z = gridder.scatter(area, 300, z=0.)
axis_ref = gm*rand()  # reference semi-axis

model = [OblateEllipsoid(-3*gm, -3*gm, 3*axis_ref,
                         0.6*axis_ref, axis_ref,
                         180*rand(), 90*rand(), 90*rand(),
                         {'k': [0.7, 0.7, 0.7, 90., 47., 13.]}),
         OblateEllipsoid(0, 0, 2*axis_ref,
                         0.5*axis_ref, axis_ref,
                         180*rand(), 90*rand(), 90*rand(),
                         {'remanence': [4., 25., 40.],
                          'k': [0.562, 0.485, 0.25, 90., 0., 0.]}),
         OblateEllipsoid(3*gm, 3*gm, 3*axis_ref,
                         0.3*axis_ref, axis_ref,
                         180*rand(), 90*rand(), 90*rand(),
                         {'remanence': [8., 25., 40.]})]


def test_oblate_ellipsoid_force_prop():
    "Test the oblate_ellipsoid code with forcing a physical property value"
    pmag = utils.ang2vec(-1, sinc, sdec)
    bx = oblate_ellipsoid.bx(x, y, z, model,
                             F, inc, dec, pmag=pmag)
    by = oblate_ellipsoid.by(x, y, z, model,
                             F, inc, dec, pmag=pmag)
    bz = oblate_ellipsoid.bz(x, y, z, model,
                             F, inc, dec, pmag=pmag)
    tf = oblate_ellipsoid.tf(x, y, z, model,
                             F, inc, dec, pmag=pmag)
    f = 1 + 4*rand()
    bx2 = oblate_ellipsoid.bx(x, y, z, model,
                              F, inc, dec, pmag=f*pmag)
    by2 = oblate_ellipsoid.by(x, y, z, model,
                              F, inc, dec, pmag=f*pmag)
    bz2 = oblate_ellipsoid.bz(x, y, z, model,
                              F, inc, dec, pmag=f*pmag)
    tf2 = oblate_ellipsoid.tf(x, y, z, model,
                              F, inc, dec, pmag=f*pmag)
    assert np.allclose(bx2, f*bx)
    assert np.allclose(by2, f*by)
    assert np.allclose(bz2, f*bz)
    assert np.allclose(tf2, f*tf)


def test_oblate_ellipsoid_ignore_none():
    "Oblate ellipsoid ignores model elements that are None"
    pmag = utils.ang2vec(1 + 10*rand(), sinc, sdec)
    model_none = model[:]
    model_none[1] = None
    bx = oblate_ellipsoid.bx(x, y, z, [model[0], model[2]],
                             F, inc, dec, pmag=pmag)
    by = oblate_ellipsoid.by(x, y, z, [model[0], model[2]],
                             F, inc, dec, pmag=pmag)
    bz = oblate_ellipsoid.bz(x, y, z, [model[0], model[2]],
                             F, inc, dec, pmag=pmag)
    tf = oblate_ellipsoid.tf(x, y, z, [model[0], model[2]],
                             F, inc, dec, pmag=pmag)
    bx2 = oblate_ellipsoid.bx(x, y, z, model_none,
                              F, inc, dec, pmag=pmag)
    by2 = oblate_ellipsoid.by(x, y, z, model_none,
                              F, inc, dec, pmag=pmag)
    bz2 = oblate_ellipsoid.bz(x, y, z, model_none,
                              F, inc, dec, pmag=pmag)
    tf2 = oblate_ellipsoid.tf(x, y, z, model_none,
                              F, inc, dec, pmag=pmag)
    assert np.allclose(bx2, bx)
    assert np.allclose(by2, by)
    assert np.allclose(bz2, bz)
    assert np.allclose(tf2, tf)


def test_oblate_ellipsoid_ignore_missing_prop():
    "Oblate ellipsoid ignores model elements that don't have \
    the needed property"
    pmag = utils.ang2vec(1 + 10*rand(), sinc, sdec)
    model_none = model[:]
    del model_none[1].props['k']
    del model_none[1].props['remanence']
    bx = oblate_ellipsoid.bx(x, y, z, [model[0], model[2]],
                             F, inc, dec, pmag=pmag)
    by = oblate_ellipsoid.by(x, y, z, [model[0], model[2]],
                             F, inc, dec, pmag=pmag)
    bz = oblate_ellipsoid.bz(x, y, z, [model[0], model[2]],
                             F, inc, dec, pmag=pmag)
    tf = oblate_ellipsoid.tf(x, y, z, [model[0], model[2]],
                             F, inc, dec, pmag=pmag)
    bx2 = oblate_ellipsoid.bx(x, y, z, model_none,
                              F, inc, dec, pmag=pmag)
    by2 = oblate_ellipsoid.by(x, y, z, model_none,
                              F, inc, dec, pmag=pmag)
    bz2 = oblate_ellipsoid.bz(x, y, z, model_none,
                              F, inc, dec, pmag=pmag)
    tf2 = oblate_ellipsoid.tf(x, y, z, model_none,
                              F, inc, dec, pmag=pmag)
    assert np.allclose(bx2, bx)
    assert np.allclose(by2, by)
    assert np.allclose(bz2, bz)
    assert np.allclose(tf2, tf)


def test_oblate_ellipsoid_demagnetizing_factors():
    "The summation of the demagnetizing factors must be equal to one"
    a = model[2].a
    b = model[2].b
    strike = model[0].strike
    dip = model[0].dip
    rake = model[0].rake
    alpha, gamma, delta = oblate_ellipsoid.structural_angles(strike, dip,
                                                             rake)
    matrix = oblate_ellipsoid.V(alpha, gamma, delta)
    x1, x2, x3 = oblate_ellipsoid.x1x2x3(x, y, z, model[2].x, model[2].y,
                                         model[2].z, matrix)
    lamb = oblate_ellipsoid._lamb(x1, x2, x3, a, b)
    denominator = oblate_ellipsoid._dlamb_aux(x1, x2, x3, a, b, lamb)
    dlamb_dx = oblate_ellipsoid._dlamb(x1, x2, x3, a, b, lamb,
                                       denominator, deriv='x')
    dlamb_dy = oblate_ellipsoid._dlamb(x1, x2, x3, a, b, lamb,
                                       denominator, deriv='y')
    dlamb_dz = oblate_ellipsoid._dlamb(x1, x2, x3, a, b, lamb,
                                       denominator, deriv='z')
    n11, n22 = oblate_ellipsoid.demag_factors(a, b)
    assert np.allclose(n11+n22+n22, 1.)


def test_oblate_ellipsoid_depolarization_tensor():
    "The depolarization tensor must be symmetric"
    a = model[0].a
    b = model[0].b
    strike = model[0].strike
    dip = model[0].dip
    rake = model[0].rake
    alpha, gamma, delta = oblate_ellipsoid.structural_angles(strike, dip,
                                                             rake)
    matrix = oblate_ellipsoid.V(alpha, gamma, delta)
    x1, x2, x3 = oblate_ellipsoid.x1x2x3(x, y, z, model[2].x, model[2].y,
                                         model[2].z, matrix)
    lamb = oblate_ellipsoid._lamb(x1, x2, x3, a, b)
    denominator = oblate_ellipsoid._dlamb_aux(x1, x2, x3, a, b, lamb)
    dlamb_dx = oblate_ellipsoid._dlamb(x1, x2, x3, a, b, lamb,
                                       denominator, deriv='x')
    dlamb_dy = oblate_ellipsoid._dlamb(x1, x2, x3, a, b, lamb,
                                       denominator, deriv='y')
    dlamb_dz = oblate_ellipsoid._dlamb(x1, x2, x3, a, b, lamb,
                                       denominator, deriv='z')
    hx = oblate_ellipsoid._hv(a, b, lamb, v='x')
    hy = oblate_ellipsoid._hv(a, b, lamb, v='y')
    hz = oblate_ellipsoid._hv(a, b, lamb, v='z')
    gx = oblate_ellipsoid._gv(a, b, lamb, v='x')
    gy = oblate_ellipsoid._gv(a, b, lamb, v='y')
    gz = oblate_ellipsoid._gv(a, b, lamb, v='z')
    # elements of the depolarization tensor without the ellipsoid
    nxx = -0.5*a*b*b*(dlamb_dx*hx*x1 + gx)
    nyy = -0.5*a*b*b*(dlamb_dy*hy*x2 + gy)
    nzz = -0.5*a*b*b*(dlamb_dz*hz*x3 + gz)
    nxy = -0.5*a*b*b*(dlamb_dx*hy*x2)
    nyx = -0.5*a*b*b*(dlamb_dy*hx*x1)
    nxz = -0.5*a*b*b*(dlamb_dx*hz*x3)
    nzx = -0.5*a*b*b*(dlamb_dz*hx*x1)
    nyz = -0.5*a*b*b*(dlamb_dy*hz*x3)
    nzy = -0.5*a*b*b*(dlamb_dz*hy*x2)
    trace = nxx+nyy+nzz
    # the atol value was found empirically
    assert np.allclose(trace, np.zeros_like(nxx), atol=0.05)
    # symmetry tests
    assert np.allclose(nxy, nyx, atol=0.05)
    assert np.allclose(nxz, nzx, atol=0.05)
    assert np.allclose(nyz, nzy, atol=0.05)


def test_oblate_ellipsoid_isotropic_susceptibility():
    "Isostropic susceptibility must be proportional to identity"
    strike = model[0].strike
    dip = model[0].dip
    rake = model[0].rake
    alpha, gamma, delta = oblate_ellipsoid.structural_angles(strike, dip,
                                                             rake)
    matrix = oblate_ellipsoid.V(alpha, gamma, delta)
    k1, k2, k3, alphas, gammas, deltas = model[0].props['k']
    suscep = oblate_ellipsoid.K(k1, k2, k3, alphas, gammas, deltas)
    assert np.allclose(suscep, k1*np.identity(3))


def test_oblate_ellipsoid_mag_isostropic_suscep():
    "Magnetization must be parallel to inducing field F"
    a = model[0].a
    b = model[0].b
    strike = model[0].strike
    dip = model[0].dip
    rake = model[0].rake
    alpha, gamma, delta = oblate_ellipsoid.structural_angles(strike, dip,
                                                             rake)
    matrix = oblate_ellipsoid.V(alpha, gamma, delta)
    k1, k2, k3, alphas, gammas, deltas = model[0].props['k']
    suscep = oblate_ellipsoid.K(k1, k2, k3, alphas, gammas, deltas)
    n11, n22 = oblate_ellipsoid.demag_factors(a, b)

    M = oblate_ellipsoid.magnetization(n11, n22, suscep, F,
                                       inc, dec, 0, 0, 0, matrix)
    H0_tilde = np.dot(matrix.T, utils.ang2vec(F/(4*np.pi*100), inc, dec))
    M_expected = np.dot(np.diag([k1/(1 - k1*n11),
                                 k2/(1 - k2*n22),
                                 k3/(1 - k3*n22)]), H0_tilde)

    assert np.allclose(M, M_expected)


def test_oblate_ellipsoid_V_orthogonal():
    pi = np.pi
    matrix = oblate_ellipsoid.V(pi*rand(), 0.5*pi*rand(), 0.5*pi*rand())
    assert np.allclose(np.dot(matrix.T, matrix), np.identity(3))
    assert np.allclose(np.dot(matrix, matrix.T), np.identity(3))


def test_oblate_ellipsoid_V_specific():
    pi = np.pi
    strike = 180.
    dip = 180.
    rake = 0.
    alpha, gamma, delta = oblate_ellipsoid.structural_angles(strike, dip,
                                                             rake)
    matrix = oblate_ellipsoid.V(alpha, gamma, delta)
    assert np.allclose(matrix, np.identity(3)[[1, 2, 0]])
