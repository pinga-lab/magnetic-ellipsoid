import numpy as np
from numpy.random import rand

from ... import utils, gridder
from ...mesher import TriaxialEllipsoid
from .. import triaxial_ellipsoid

F = 20000 + 40000*rand()
inc = -90 + 180*rand()
dec = 180*rand()
sinc = -90 + 180*rand()
sdec = 180*rand()

gm = 1e4*rand()  # geometrical factor
area = [-5.*gm, 5.*gm, -5.*gm, 5.*gm]
x, y, z = gridder.scatter(area, 300, z=0.)
axis_ref = gm*rand()  # reference semi-axis

model = [TriaxialEllipsoid(-3*gm, -3*gm, 3*axis_ref,
                           axis_ref, 0.8*axis_ref, 0.6*axis_ref,
                           180*rand(), 90*rand(), 90*rand(),
                           {'k': [0.7, 0.7, 0.7, 90., 47., 13.]}),
         TriaxialEllipsoid(0, 0, 2*axis_ref,
                           axis_ref, 0.8*axis_ref, 0.6*axis_ref,
                           180*rand(), 90*rand(), 90*rand(),
                           {'remanence': [4., 25., 40.],
                            'k': [0.562, 0.485, 0.25, 90., 0., 0.]}),
         TriaxialEllipsoid(3*gm, 3*gm, 3*axis_ref,
                           axis_ref, 0.8*axis_ref, 0.6*axis_ref,
                           180*rand(), 90*rand(), 90*rand(),
                           {'remanence': [8., 25., 40.]})]


def test_triaxial_ellipsoid_force_prop():
    "Test the triaxial_ellipsoid code with forcing a physical property value"
    pmag = utils.ang2vec(-1, sinc, sdec)
    bx = triaxial_ellipsoid.bx(x, y, z, model,
                               F, inc, dec, pmag=pmag)
    by = triaxial_ellipsoid.by(x, y, z, model,
                               F, inc, dec, pmag=pmag)
    bz = triaxial_ellipsoid.bz(x, y, z, model,
                               F, inc, dec, pmag=pmag)
    tf = triaxial_ellipsoid.tf(x, y, z, model,
                               F, inc, dec, pmag=pmag)
    f = 1 + 4*rand()
    bx2 = triaxial_ellipsoid.bx(x, y, z, model,
                                F, inc, dec, pmag=f*pmag)
    by2 = triaxial_ellipsoid.by(x, y, z, model,
                                F, inc, dec, pmag=f*pmag)
    bz2 = triaxial_ellipsoid.bz(x, y, z, model,
                                F, inc, dec, pmag=f*pmag)
    tf2 = triaxial_ellipsoid.tf(x, y, z, model,
                                F, inc, dec, pmag=f*pmag)
    assert np.allclose(bx2, f*bx)
    assert np.allclose(by2, f*by)
    assert np.allclose(bz2, f*bz)
    assert np.allclose(tf2, f*tf)


def test_triaxial_ellipsoid_ignore_none():
    "Triaxial ellipsoid ignores model elements that are None"
    pmag = utils.ang2vec(1 + 10*rand(), sinc, sdec)
    model_none = model[:]
    model_none[1] = None
    bx = triaxial_ellipsoid.bx(x, y, z, [model[0], model[2]],
                               F, inc, dec, pmag=pmag)
    by = triaxial_ellipsoid.by(x, y, z, [model[0], model[2]],
                               F, inc, dec, pmag=pmag)
    bz = triaxial_ellipsoid.bz(x, y, z, [model[0], model[2]],
                               F, inc, dec, pmag=pmag)
    tf = triaxial_ellipsoid.tf(x, y, z, [model[0], model[2]],
                               F, inc, dec, pmag=pmag)
    bx2 = triaxial_ellipsoid.bx(x, y, z, model_none,
                                F, inc, dec, pmag=pmag)
    by2 = triaxial_ellipsoid.by(x, y, z, model_none,
                                F, inc, dec, pmag=pmag)
    bz2 = triaxial_ellipsoid.bz(x, y, z, model_none,
                                F, inc, dec, pmag=pmag)
    tf2 = triaxial_ellipsoid.tf(x, y, z, model_none,
                                F, inc, dec, pmag=pmag)
    assert np.allclose(bx2, bx)
    assert np.allclose(by2, by)
    assert np.allclose(bz2, bz)
    assert np.allclose(tf2, tf)


def test_triaxial_ellipsoid_ignore_missing_prop():
    "Triaxial ellipsoid ignores model elements that don't have \
    the needed property"
    pmag = utils.ang2vec(1 + 10*rand(), sinc, sdec)
    model_none = model[:]
    del model_none[1].props['k']
    del model_none[1].props['remanence']
    bx = triaxial_ellipsoid.bx(x, y, z, [model[0], model[2]],
                               F, inc, dec, pmag=pmag)
    by = triaxial_ellipsoid.by(x, y, z, [model[0], model[2]],
                               F, inc, dec, pmag=pmag)
    bz = triaxial_ellipsoid.bz(x, y, z, [model[0], model[2]],
                               F, inc, dec, pmag=pmag)
    tf = triaxial_ellipsoid.tf(x, y, z, [model[0], model[2]],
                               F, inc, dec, pmag=pmag)
    bx2 = triaxial_ellipsoid.bx(x, y, z, model_none,
                                F, inc, dec, pmag=pmag)
    by2 = triaxial_ellipsoid.by(x, y, z, model_none,
                                F, inc, dec, pmag=pmag)
    bz2 = triaxial_ellipsoid.bz(x, y, z, model_none,
                                F, inc, dec, pmag=pmag)
    tf2 = triaxial_ellipsoid.tf(x, y, z, model_none,
                                F, inc, dec, pmag=pmag)
    assert np.allclose(bx2, bx)
    assert np.allclose(by2, by)
    assert np.allclose(bz2, bz)
    assert np.allclose(tf2, tf)


def test_triaxial_ellipsoid_demagnetizing_factors():
    "The summation of the demagnetizing factors must be equal to one"
    a = model[2].a
    b = model[2].b
    c = model[2].c
    strike = model[0].strike
    dip = model[0].dip
    rake = model[0].rake
    alpha, gamma, delta = triaxial_ellipsoid.structural_angles(strike, dip,
                                                               rake)
    matrix = triaxial_ellipsoid.V(alpha, gamma, delta)
    x1, x2, x3 = triaxial_ellipsoid.x1x2x3(x, y, z, model[2].x, model[2].y,
                                           model[2].z, matrix)
    lamb = triaxial_ellipsoid._lamb(x1, x2, x3, a, b, c)
    denominator = triaxial_ellipsoid._dlamb_aux(x1, x2, x3, a, b, c, lamb)
    dlamb_dx = triaxial_ellipsoid._dlamb(x1, x2, x3, a, b, c, lamb,
                                         denominator, deriv='x')
    dlamb_dy = triaxial_ellipsoid._dlamb(x1, x2, x3, a, b, c, lamb,
                                         denominator, deriv='y')
    dlamb_dz = triaxial_ellipsoid._dlamb(x1, x2, x3, a, b, c, lamb,
                                         denominator, deriv='z')
    n11, n22, n33 = triaxial_ellipsoid.demag_factors(a, b, c)
    assert np.allclose(n11+n22+n33, 1.)


def test_triaxial_self_demagnetization():
    "Self-demagnetization decreases the magnetization intensity"
    a = model[0].a
    b = model[0].b
    c = model[0].c
    strike = model[0].strike
    dip = model[0].dip
    rake = model[0].rake
    alpha, gamma, delta = triaxial_ellipsoid.structural_angles(strike, dip,
                                                               rake)
    matrix = triaxial_ellipsoid.V(alpha, gamma, delta)
    k1, k2, k3, alphas, gammas, deltas = model[0].props['k']
    suscep = triaxial_ellipsoid.K(k1, k2, k3, alphas, gammas, deltas)
    n11, n22, n33 = triaxial_ellipsoid.demag_factors(a, b, c)

    # magnetization with self-demagnetization
    axes = [a, b, c]
    M = triaxial_ellipsoid.magnetization(suscep, F, inc, dec, 0, 0, 0,
                                         matrix, axes)

    # magnetization without self-demagnetization
    M_approx = triaxial_ellipsoid.magnetization(suscep, F, inc, dec, 0, 0, 0,
                                                matrix, axes=None)

    M_norm = np.linalg.norm(M, ord=2)
    M_approx_norm = np.linalg.norm(M_approx, ord=2)
    assert M_norm < M_approx_norm


def test_triaxial_ellipsoid_depolarization_tensor():
    "The depolarization tensor must be symmetric"
    a = model[0].a
    b = model[0].b
    c = model[0].c
    strike = model[0].strike
    dip = model[0].dip
    rake = model[0].rake
    alpha, gamma, delta = triaxial_ellipsoid.structural_angles(strike, dip,
                                                               rake)
    matrix = triaxial_ellipsoid.V(alpha, gamma, delta)
    x1, x2, x3 = triaxial_ellipsoid.x1x2x3(x, y, z, model[2].x, model[2].y,
                                           model[2].z, matrix)
    lamb = triaxial_ellipsoid._lamb(x1, x2, x3, a, b, c)
    denominator = triaxial_ellipsoid._dlamb_aux(x1, x2, x3, a, b, c, lamb)
    dlamb_dx = triaxial_ellipsoid._dlamb(x1, x2, x3, a, b, c, lamb,
                                         denominator, deriv='x')
    dlamb_dy = triaxial_ellipsoid._dlamb(x1, x2, x3, a, b, c, lamb,
                                         denominator, deriv='y')
    dlamb_dz = triaxial_ellipsoid._dlamb(x1, x2, x3, a, b, c, lamb,
                                         denominator, deriv='z')
    hx = triaxial_ellipsoid._hv(a, b, c, lamb, v='x')
    hy = triaxial_ellipsoid._hv(a, b, c, lamb, v='y')
    hz = triaxial_ellipsoid._hv(a, b, c, lamb, v='z')
    kappa, phi = triaxial_ellipsoid._E_F_field_args(a, b, c, lamb)
    gx = triaxial_ellipsoid._gv(a, b, c, kappa, phi, v='x')
    gy = triaxial_ellipsoid._gv(a, b, c, kappa, phi, v='y')
    gz = triaxial_ellipsoid._gv(a, b, c, kappa, phi, v='z')
    # elements of the depolarization tensor without the ellipsoid
    nxx = -0.5*a*b*c*(dlamb_dx*hx*x1 + gx)
    nyy = -0.5*a*b*c*(dlamb_dy*hy*x2 + gy)
    nzz = -0.5*a*b*c*(dlamb_dz*hz*x3 + gz)
    nxy = -0.5*a*b*c*(dlamb_dx*hy*x2)
    nyx = -0.5*a*b*c*(dlamb_dy*hx*x1)
    nxz = -0.5*a*b*c*(dlamb_dx*hz*x3)
    nzx = -0.5*a*b*c*(dlamb_dz*hx*x1)
    nyz = -0.5*a*b*c*(dlamb_dy*hz*x3)
    nzy = -0.5*a*b*c*(dlamb_dz*hy*x2)
    trace = nxx+nyy+nzz
    # the atol value was found empirically
    assert np.allclose(trace, np.zeros_like(nxx), atol=0.05)
    # symmetry tests
    assert np.allclose(nxy, nyx)
    assert np.allclose(nxz, nzx)
    assert np.allclose(nyz, nzy)


def test_triaxial_ellipsoid_isotropic_susceptibility():
    "Isostropic susceptibility must be proportional to identity"
    strike = model[0].strike
    dip = model[0].dip
    rake = model[0].rake
    alpha, gamma, delta = triaxial_ellipsoid.structural_angles(strike, dip,
                                                               rake)
    matrix = triaxial_ellipsoid.V(alpha, gamma, delta)
    k1, k2, k3, alphas, gammas, deltas = model[0].props['k']
    suscep = triaxial_ellipsoid.K(k1, k2, k3, alphas, gammas, deltas)
    assert np.allclose(suscep, k1*np.identity(3))


def test_triaxial_ellipsoid_mag_isostropic_suscep():
    "Magnetization assumes a simple form"
    a = model[0].a
    b = model[0].b
    c = model[0].c
    strike = model[0].strike
    dip = model[0].dip
    rake = model[0].rake
    alpha, gamma, delta = triaxial_ellipsoid.structural_angles(strike, dip,
                                                               rake)
    matrix = triaxial_ellipsoid.V(alpha, gamma, delta)
    k1, k2, k3, alphas, gammas, deltas = model[0].props['k']
    suscep = triaxial_ellipsoid.K(k1, k2, k3, alphas, gammas, deltas)
    n11, n22, n33 = triaxial_ellipsoid.demag_factors(a, b, c)

    axes = [a, b, c]
    M = triaxial_ellipsoid.magnetization(suscep, F, inc, dec, 0, 0, 0, matrix,
                                         axes)
    H0 = utils.ang2vec(F/(4*np.pi*100), inc, dec)
    suscep_tilde = np.dot(np.dot(matrix.T, suscep), matrix)
    aux = np.linalg.inv(np.identity(3) + np.dot(suscep_tilde,
                                                np.diag([n11, n22, n33])))
    Lambda = np.dot(np.dot(matrix, aux), matrix.T)
    M_expected = np.dot(Lambda, np.dot(suscep, H0))

    assert np.allclose(M, M_expected)


def test_triaxial_ellipsoid_V_orthogonal():
    a = 1000.*rand()
    b = 0.8*a
    c = 0.6*a
    pi = np.pi
    matrix = triaxial_ellipsoid.V(pi*rand(), 0.5*pi*rand(), 0.5*pi*rand())
    assert np.allclose(np.dot(matrix.T, matrix), np.identity(3))
    assert np.allclose(np.dot(matrix, matrix.T), np.identity(3))


def test_triaxial_ellipsoid_V_identity():
    a = 1000.*rand()
    b = 0.8*a
    c = 0.6*a
    pi = np.pi
    strike = 180.
    dip = 180.
    rake = 0.
    alpha, gamma, delta = triaxial_ellipsoid.structural_angles(strike, dip,
                                                               rake)
    matrix = triaxial_ellipsoid.V(alpha, gamma, delta)
    assert np.allclose(matrix, np.identity(3))
