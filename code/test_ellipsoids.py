from __future__ import division, absolute_import
from mesher import TriaxialEllipsoid
from mesher import ProlateEllipsoid
from mesher import OblateEllipsoid
from mesher import _coord_transf_matrix_triaxial
from mesher import _coord_transf_matrix_oblate
from mesher import _multi_dot, _R1, _R2, _R3
import numpy as np
from numpy.testing import assert_almost_equal
from pytest import raises


def test_triaxial_ellipsoid_copy():
    'Check the elements of a duplicated ellipsoid'
    orig = TriaxialEllipsoid(12, 42, 53, 61, 35, 14, 10, 20, 30, {
                             'remanent magnetization': [10, 25, 40],
                             'principal susceptibilities': [0.562, 0.485,
                                                            0.25],
                             'susceptibility angles': [90, 0, 0]})
    cp = orig.copy()
    assert orig is not cp
    assert orig.x == cp.x
    assert orig.y == cp.y
    assert orig.z == cp.z
    assert orig.large_axis == cp.large_axis
    assert orig.intermediate_axis == cp.intermediate_axis
    assert orig.small_axis == cp.small_axis
    assert orig.props == cp.props

    cp.x = 4
    cp.y = 6
    cp.z = 7
    cp.props['principal susceptibilities'] = [0.7, 0.9, 1]
    assert orig.x != cp.x
    assert orig.y != cp.y
    assert orig.z != cp.z
    assert orig.props['principal susceptibilities'] != \
        cp.props['principal susceptibilities']


def test_triaxial_ellipsoid_axes():
    'axes must be given in descending order'
    raises(AssertionError, TriaxialEllipsoid, x=1, y=2, z=3,
           large_axis=6, intermediate_axis=5, small_axis=14,
           strike=10, dip=20, rake=30)


def test_triaxial_ellipsoid_principal_susceptibilities_fmt():
    'principal susceptibilities must be a list containing 3 elements'
    e = TriaxialEllipsoid(x=1, y=2, z=3, large_axis=6,
                          intermediate_axis=5, small_axis=4,
                          strike=10, dip=20, rake=30,
                          props={'remanent magnetization': [10, 25, 40],
                                 'principal susceptibilities': [0.562, 0.485],
                                 'susceptibility angles': [90, 0, 0]})

    with raises(AssertionError):
        e.susceptibility_tensor


def test_triaxial_ellipsoid_susceptibility_angles_fmt():
    'susceptibility angles must be a list containing 3 elements'
    e = TriaxialEllipsoid(x=1, y=2, z=3, large_axis=6,
                          intermediate_axis=5, small_axis=4,
                          strike=10, dip=20, rake=30,
                          props={'remanent magnetization': [10, 25, 40],
                                 'principal susceptibilities': [0.562, 0.485,
                                                                0.2],
                                 'susceptibility angles': [90, 0]})

    with raises(AssertionError):
        e.susceptibility_tensor


def test_triaxial_ellipsoid_susceptibility_tensor_symm():
    'susceptibility tensor must be symmetric'
    e = TriaxialEllipsoid(x=1, y=2, z=3, large_axis=6,
                          intermediate_axis=5, small_axis=4,
                          strike=10, dip=20, rake=30,
                          props={'remanent magnetization': [10, 25, 40],
                                 'principal susceptibilities': [0.562, 0.485,
                                                                0.25],
                                 'susceptibility angles': [90, 0, 0]})
    assert_almost_equal(e.susceptibility_tensor, e.susceptibility_tensor.T,
                        decimal=15)


def test_triaxial_ellipsoid_principal_susceptibilities_order():
    'principal susceptibilities must be given in descending order'
    e = TriaxialEllipsoid(x=1, y=2, z=3, large_axis=6,
                          intermediate_axis=5, small_axis=4,
                          strike=10, dip=20, rake=30,
                          props={'remanent magnetization': [10, 25, 40],
                                 'principal susceptibilities': [0.562, 0.185,
                                                                0.25],
                                 'susceptibility angles': [90, 0, 0]})
    with raises(AssertionError):
        e.susceptibility_tensor


def test_triaxial_ellipsoid_principal_susceptibilities_signal():
    'principal susceptibilities must be all positive'
    e = TriaxialEllipsoid(x=1, y=2, z=3, large_axis=6,
                          intermediate_axis=5, small_axis=4,
                          strike=10, dip=20, rake=30,
                          props={'remanent magnetization': [10, 25, 40],
                                 'principal susceptibilities': [0.562, 0.485,
                                                                -0.25],
                                 'susceptibility angles': [90, 0, 0]})
    with raises(AssertionError):
        e.susceptibility_tensor


def test_coord_transf_matrix_triaxial_known():
    'Coordinate transformation matrix built with known orientation angles'
    strike = 0
    dip = 0
    rake = 0
    transf_matrix = _coord_transf_matrix_triaxial(strike, dip, rake)
    benchmark = np.array([[1, 0, 0],
                          [0, -1, 0],
                          [0, 0, -1]])
    assert_almost_equal(transf_matrix, benchmark, decimal=15)


def test_coord_transf_matrix_triaxial_orthogonal():
    'Coordinate transformation matrix must be orthogonal'
    strike = 38.9
    dip = -0.2
    rake = 174
    transf_matrix = _coord_transf_matrix_triaxial(strike, dip, rake)
    dot1 = np.dot(transf_matrix, transf_matrix.T)
    dot2 = np.dot(transf_matrix.T, transf_matrix)
    assert_almost_equal(dot1, dot2, decimal=15)
    assert_almost_equal(dot1, np.identity(3), decimal=15)
    assert_almost_equal(dot2, np.identity(3), decimal=15)


def test_prolate_ellipsoid_copy():
    'Check the elements of a duplicated ellipsoid'
    orig = ProlateEllipsoid(31, 2, 83, 56, 54, 1, 29, 70,
                            props={'remanent magnetization': [10, 25, 40],
                                   'principal susceptibilities': [0.562, 0.485,
                                                                  -0.25],
                                   'susceptibility angles': [90, 0, 0]})
    cp = orig.copy()
    assert orig is not cp
    assert orig.x == cp.x
    assert orig.y == cp.y
    assert orig.z == cp.z
    assert orig.large_axis == cp.large_axis
    assert orig.small_axis == cp.small_axis
    assert orig.props == cp.props

    cp.x = 4
    cp.y = 6
    cp.z = 7
    cp.props['principal susceptibilities'] = [0.7, 0.9, 1]
    assert orig.x != cp.x
    assert orig.y != cp.y
    assert orig.z != cp.z
    assert orig.props['principal susceptibilities'] != \
        cp.props['principal susceptibilities']


def test_prolate_ellipsoid_axes():
    'axes must be given in descending order'
    raises(AssertionError, ProlateEllipsoid, x=1, y=2, z=3,
           large_axis=2, small_axis=4, strike=10, dip=20, rake=30)


def test_prolate_ellipsoid_principal_susceptibilities_fmt():
    'principal susceptibilities must be a list containing 3 elements'
    e = ProlateEllipsoid(x=1, y=2, z=3,
                         large_axis=6, small_axis=4,
                         strike=10, dip=20, rake=30,
                         props={'remanent magnetization': [10, 25, 40],
                                'principal susceptibilities': [0.562, 0.485],
                                'susceptibility angles': [90, 0, 0]})

    with raises(AssertionError):
        e.susceptibility_tensor


def test_prolate_ellipsoid_susceptibility_angles_fmt():
    'susceptibility angles must be a list containing 3 elements'
    e = ProlateEllipsoid(x=1, y=2, z=3,
                         large_axis=6, small_axis=4,
                         strike=10, dip=20, rake=30,
                         props={'remanent magnetization': [10, 25, 40],
                                'principal susceptibilities': [0.562, 0.485,
                                                               0.2],
                                'susceptibility angles': [90, 0]})

    with raises(AssertionError):
        e.susceptibility_tensor


def test_prolate_ellipsoid_principal_susceptibilities_order():
    'principal susceptibilities must be given in descending order'
    e = ProlateEllipsoid(x=1, y=2, z=3, large_axis=6, small_axis=4,
                         strike=10, dip=20, rake=30,
                         props={'remanent magnetization': [10, 25, 40],
                                'principal susceptibilities': [0.562, 0.85,
                                                               0.2],
                                'susceptibility angles': [90, 0]})
    with raises(AssertionError):
        e.susceptibility_tensor


def test_prolate_ellipsoid_principal_susceptibilities_signal():
    'principal susceptibilities must be all positive'
    e = ProlateEllipsoid(x=1, y=2, z=3, large_axis=6, small_axis=4,
                         strike=10, dip=20, rake=30,
                         props={'remanent magnetization': [10, 25, 40],
                                'principal susceptibilities': [0.562, 0.485,
                                                               -0.2],
                                'susceptibility angles': [90, 0]})
    with raises(AssertionError):
        e.susceptibility_tensor


def test_oblate_ellipsoid_copy():
    'Check the elements of a duplicated ellipsoid'
    orig = OblateEllipsoid(1, 2, 3, 4, 6, 10, 20, 30, {
        'remanent magnetization': [3, -2, 40],
        'susceptibility tensor': [0.562, 0.485, 0.25,
                                  90, 34, 0]})
    cp = orig.copy()
    assert orig is not cp
    assert orig.x == cp.x
    assert orig.y == cp.y
    assert orig.z == cp.z
    assert orig.small_axis == cp.small_axis
    assert orig.large_axis == cp.large_axis
    assert orig.props == cp.props

    cp.x = 4
    cp.y = 6
    cp.z = 7
    cp.props['remanent magnetization'] = [100, -40, -25]
    assert orig.x != cp.x
    assert orig.y != cp.y
    assert orig.z != cp.z
    assert orig.props['remanent magnetization'] != \
        cp.props['remanent magnetization']


def test_oblate_ellipsoid_axes():
    'axes must be given in ascending order'
    raises(AssertionError, OblateEllipsoid, x=1, y=2, z=3,
           small_axis=12, large_axis=4, strike=10, dip=20, rake=30)


def test_oblate_ellipsoid_principal_susceptibilities_fmt():
    'principal susceptibilities must be a list containing 3 elements'
    e = OblateEllipsoid(x=1, y=2, z=3,
                        small_axis=4, large_axis=6,
                        strike=10, dip=20, rake=30,
                        props={'remanent magnetization': [10, 25, 40],
                               'principal susceptibilities': [0.562, 0.485],
                               'susceptibility angles': [90, 0, 0]})

    with raises(AssertionError):
        e.susceptibility_tensor


def test_oblate_ellipsoid_susceptibility_angles_fmt():
    'susceptibility angles must be a list containing 3 elements'
    e = OblateEllipsoid(x=1, y=2, z=3,
                        small_axis=4, large_axis=6,
                        strike=10, dip=20, rake=30,
                        props={'remanent magnetization': [10, 25, 40],
                               'principal susceptibilities': [0.562, 0.485,
                                                              0.2],
                               'susceptibility angles': [90, 0]})

    with raises(AssertionError):
        e.susceptibility_tensor


def test_oblate_ellipsoid_susceptibility_tensor_symm():
    'susceptibility tensor must be symmetric'
    e = OblateEllipsoid(x=1, y=2, z=3,
                        small_axis=4, large_axis=6,
                        strike=10, dip=20, rake=30,
                        props={'remanent magnetization': [10, 25, 40],
                               'principal susceptibilities': [0.562, 0.485,
                                                              0.25],
                               'susceptibility angles': [-240, 71, -2]})
    assert_almost_equal(e.susceptibility_tensor, e.susceptibility_tensor.T,
                        decimal=15)


def test_oblate_ellipsoid_principal_susceptibilities_order():
    'principal susceptibilities must be given in descending order'
    e = OblateEllipsoid(x=1, y=2, z=3, small_axis=4, large_axis=6,
                        strike=10, dip=20, rake=30,
                        props={'remanent magnetization': [10, 25, 40],
                               'principal susceptibilities': [0.562, 0.485,
                                                              0.9],
                               'susceptibility angles': [19, -14, 100]})
    with raises(AssertionError):
        e.susceptibility_tensor


def test_oblate_ellipsoid_principal_susceptibilities_signal():
    'principal susceptibilities must be all positive'
    e = OblateEllipsoid(x=1, y=2, z=3, small_axis=4, large_axis=6,
                        strike=10, dip=20, rake=30,
                        props={'remanent magnetization': [10, 25, 40],
                               'principal susceptibilities': [0.562, 0.485,
                                                              0.9],
                               'susceptibility angles': [19, -14, 100]})
    with raises(AssertionError):
        e.susceptibility_tensor


def test_coord_transf_matrix_oblate_known():
    'Coordinate transformation matrix built with known orientation angles'
    strike = 0
    dip = 0
    rake = 0
    transf_matrix = _coord_transf_matrix_oblate(strike, dip, rake)
    benchmark = np.array([[0, 1, 0],
                          [0, 0, -1],
                          [-1, 0, 0]])
    assert_almost_equal(transf_matrix, benchmark, decimal=15)


def test_coord_transf_matrix_oblate_orthonal():
    'Coordinate transformation matrix must be orthogonal'
    strike = 7
    dip = 23
    rake = -np.pi/3
    transf_matrix = _coord_transf_matrix_oblate(strike, dip, rake)
    dot1 = np.dot(transf_matrix, transf_matrix.T)
    dot2 = np.dot(transf_matrix.T, transf_matrix)
    assert_almost_equal(dot1, dot2, decimal=15)
    assert_almost_equal(dot1, np.identity(3), decimal=15)
    assert_almost_equal(dot2, np.identity(3), decimal=15)


def test_multi_dot_numpy_dot():
    'Compare the multi_dot with the nested numpy.dot'
    A = _R1(-87)
    B = _R2(1)
    C = _R3(24)

    assert_almost_equal(_multi_dot([A, B, C]),
                        np.dot(A, np.dot(B, C)), decimal=15)
    assert_almost_equal(_multi_dot([A, B, C]),
                        np.dot(np.dot(A, B), C), decimal=15)


def test_multi_dot_bad_arguments():
    'multi_dot fails for matrices with different sizes'
    A = _R1(7)
    B = _R2(-10)
    C = _R3(3)

    raises(AssertionError, _multi_dot, [A, B, C, np.identity(4)])
    raises(AssertionError, _multi_dot, [A, B, np.identity(5), C])
    raises(AssertionError, _multi_dot, [A, np.identity(6), B, C])


def test_R1_R2_R3_orthonal():
    'Rotation matrices must be orthogonal'
    A = _R1(-19)
    B = _R2(34.71)
    C = _R3(28)

    assert_almost_equal(np.dot(A, A.T), np.dot(A.T, A), decimal=15)
    assert_almost_equal(np.dot(A, A.T), np.identity(3), decimal=15)
    assert_almost_equal(np.dot(A.T, A), np.identity(3), decimal=15)

    assert_almost_equal(np.dot(B, B.T), np.dot(B.T, B), decimal=15)
    assert_almost_equal(np.dot(B, B.T), np.identity(3), decimal=15)
    assert_almost_equal(np.dot(B.T, B), np.identity(3), decimal=15)

    assert_almost_equal(np.dot(C, C.T), np.dot(C.T, C), decimal=15)
    assert_almost_equal(np.dot(C, C.T), np.identity(3), decimal=15)
    assert_almost_equal(np.dot(C.T, C), np.identity(3), decimal=15)
