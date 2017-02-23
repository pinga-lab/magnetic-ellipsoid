from fatiando.mesher import OblateEllipsoid
import numpy as np
from numpy.random import rand


def test_oblate_ellipsoid_copy():
    orig = OblateEllipsoid(1, 2, 3, 4, 6, 10, 20, 30, {
        'remanence': [10., 25., 40.],
        'k': [0.562, 0.485, 0.25, 90., 34., 0.]})
    cp = orig.copy()
    assert orig is not cp
    assert orig.x == cp.x
    assert orig.y == cp.y
    assert orig.z == cp.z
    assert orig.a == cp.a
    assert orig.b == cp.b
    assert orig.props == cp.props

    cp.x = 4
    cp.y = 6
    cp.z = 7
    cp.props['remanence'] = [100., -40., -25.]
    assert orig.x != cp.x
    assert orig.y != cp.y
    assert orig.z != cp.z
    assert orig.props['remanence'] != cp.props['remanence']
