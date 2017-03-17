"""
Generate and operate on various kinds of meshes and geometric elements
"""
from __future__ import division, absolute_import
from future.builtins import range, super
import numpy
import scipy.special
import scipy.interpolate
import copy as cp

from fatiando import gridder


class GeometricElement(object):

    """
    Base class for all geometric elements.
    """

    def __init__(self, props):
        self.props = {}
        if props is not None:
            for p in props:
                self.props[p] = props[p]

    def addprop(self, prop, value):
        """
        Add a physical property to this geometric element.

        If it already has the property, the given value will overwrite the
        existing one.

        Parameters:

        * prop : str
            Name of the physical property.
        * value : float
            The value of this physical property.

        """
        self.props[prop] = value

    def copy(self):
        """ Return a deep copy of the current instance."""
        return cp.deepcopy(self)


class TriaxialEllipsoid(GeometricElement):
    """
    Create an arbitrarily-oriented triaxial ellipsoid.

    Triaxial ellipsoids are those having different semi-axes.
    This code follows Clark et al. (1986) and defines the spatial
    orientation of the ellipsoid by using three angles: strike,
    rake and dip. These angles are commonly used to define the
    orientation of geological structures (Allmendinger et al., 2012).

    References:

    Allmendinger, R., Cardozo, N., and Fisher, D. M.: Structural geology
    algorithms : vectors and tensors, Cambridge University Press, 2012.

    Clark, D., Saul, S., and Emerson, D.: Magnetic and gravity anomalies
    of a triaxial ellipsoid, Exploration Geophysics, 17, 189-200, 1986.

    .. note:: The coordinate system used is x -> North, y -> East
    and z -> Down

    Parameters:

    * x, y, z : float
        The coordinates of the center of the ellipsoid.
    * large_axis, intermediate_axis, small_axis: float
        Semi-axes forming the ellipsoid (in m).
    * strike, dip, rake
        Orientation angles of the ellipsoid (in degrees).
    * props : dict
        Physical properties assigned to the ellipsoid.
        Ex: ``props={'density':10,
                     'remanent magnetization':[10, 25, 40],
                     'susceptibility tensor':[0.562, 0.485, 0.25,
                                              90, 0, 0]}``

    Examples:

        >>> e = TriaxialEllipsoid(x=1, y=2, z=3, large_axis=6,
        ...                       intermediate_axis=5, small_axis=4,
        ...                       strike=10, dip=20, rake=30, props={
        ...                       'remanent magnetization': [10, 25, 40],
        ...                       'susceptibility tensor': [0.562, 0.485,
        ...                                                 0.25, 90, 0,
        ...                                                 0]})
        >>> e.props['remanent magnetization']
        [10, 25, 40]
        >>> e.addprop('density', 20)
        >>> print(e.props['density'])
        20
        >>> print(e)
        x:1 | y:2 | z:3 | large_axis:6 | intermediate_axis:5 | small_axis:4 | \
strike:10 | dip:20 | rake:30 | density:20 | remanent magnetization:[10, 25, 40\
] | susceptibility tensor:[0.562, 0.485, 0.25, 90, 0, 0]
        >>> e = TriaxialEllipsoid(1, 2.7, 3, 6, 5, 4, 10, 20, 30)
        >>> print(e)
        x:1 | y:2.7 | z:3 | large_axis:6 | intermediate_axis:5 | small_axis:4 \
| strike:10 | dip:20 | rake:30
        >>> e.addprop('density', 2670)
        >>> print(e)
        x:1 | y:2.7 | z:3 | large_axis:6 | intermediate_axis:5 | small_axis:4 \
| strike:10 | dip:20 | rake:30 | density:2670

    """

    def __init__(self, x, y, z, large_axis, intermediate_axis, small_axis,
                 strike, dip, rake, props=None):
        super().__init__(props)

        self.x = x
        self.y = y
        self.z = z
        self.large_axis = large_axis
        self.intermediate_axis = intermediate_axis
        self.small_axis = small_axis
        self.strike = strike
        self.dip = dip
        self.rake = rake

        assert self.large_axis > self.intermediate_axis and \
            self.intermediate_axis > self.small_axis, "large_axis must be grea\
ter than intermediate_axis and intermediate_axis must greater than small_axis"

        # Auxiliary orientation angles
        alpha, gamma, delta = _auxiliary_angles(self.strike,
                                                self.dip,
                                                self.rake)

        # Coordinate transformation matrix
        self.transf_matrix = _coord_transf_matrix_triaxial(alpha,
                                                           gamma,
                                                           delta)

    def __str__(self):
        """
        Return a string representation of the triaxial ellipsoid.
        """

        names = [('x', self.x), ('y', self.y), ('z', self.z),
                 ('large_axis', self.large_axis),
                 ('intermediate_axis', self.intermediate_axis),
                 ('small_axis', self.small_axis), ('strike', self.strike),
                 ('dip', self.dip), ('rake', self.rake)]
        names = names + [(p, self.props[p]) for p in sorted(self.props)]
        return ' | '.join('%s:%s' % (n, v) for n, v in names)

    @property
    def susceptibility_tensor(self):
        '''
        Calculate the susceptibility tensor (in SI) in the main system.

        The susceptibility tensor is calculated if
        'principal susceptibilities' and 'susceptibility angles' are
        defined in the dictionary of physical properties props.
        The 'principal susceptibilities' must be a list containing
        the three positive eigenvalues (principal susceptibilities k1, k2
        and k3) of the susceptibility tensor, in descending order. The
        'susceptibility angles' must be a list containing three angles
        (in degree) used to compute the eigenvector matrix U of the
        susceptibility tensor. The eigenvector matrix is defined by the
        function _coord_transf_matrix_triaxial.
        '''

        if 'principal susceptibilities' and 'susceptibility angles' \
                in self.props:

            assert len(self.props['principal susceptibilities']) == 3, \
                'there must be three principal susceptibilities'
            assert len(self.props['susceptibility angles']) == 3, \
                'there must be three angles'

            # Large, intermediate and small eigenvalues of the
            # susceptibility tensor (principal susceptibilities)
            k1 = self.props['principal susceptibilities'][0]
            k2 = self.props['principal susceptibilities'][1]
            k3 = self.props['principal susceptibilities'][2]

            assert k1 >= k2 >= k3, 'the principal susceptibilities must be \
given in descending order'

            assert (k1 > 0) and (k2 > 0) and (k3 > 0), 'the principal \
susceptibilities must be all positive'

            # Angles (in degrees) defining the eigenvector matrix
            # of the susceptibility tensor
            strike = self.props['susceptibility angles'][0]
            dip = self.props['susceptibility angles'][1]
            rake = self.props['susceptibility angles'][2]

            # Eigenvector matrix of the susceptibility tensor
            alpha, gamma, delta = _auxiliary_angles(strike, dip, rake)
            U = _coord_transf_matrix_triaxial(alpha, gamma, delta)

            suscep_tensor = numpy.dot(U, numpy.diag([k1, k2, k3]))
            suscep_tensor = numpy.dot(suscep_tensor, U.T)

            return suscep_tensor

        else:
            return None


class ProlateEllipsoid(GeometricElement):
    """
    Create an arbitrarily-oriented prolate ellipsoid.

    Prolate ellipsoids are those having symmetry around the large axes.
    This code follows Emerson et al. (1985) and defines the spatial
    orientation of the ellipsoid by using three angles: strike,
    rake and dip. These angles are commonly used to define the
    orientation of geological structures (Allmendinger et al., 2012).

    References:

    Allmendinger, R., Cardozo, N., and Fisher, D. M.: Structural geology
    algorithms : vectors and tensors, Cambridge University Press, 2012.

    Emerson, D. W., Clark, D., and Saul, S.: Magnetic exploration models
    incorporating remanence, demagnetization and anisotropy: HP 41C
    handheld computer algorithms, Exploration Geophysics, 16, 1-122, 1985.

    .. note:: The coordinate system used is x -> North, y -> East
    and z -> Down

    Parameters:

    * x, y, z : float
        The coordinates of the center of the ellipsoid.
    * large_axis, small_axis: float
        Semi-axes forming the ellipsoid.
    * strike, dip, rake
        Orientation angles of the ellipsoid.
    * props : dict
        Physical properties assigned to the ellipsoid.
        Ex: ``props={'density':10,
                     'remanent magnetization':[13, -5, 7.4],
                     'susceptibility tensor':[0.562, 0.485, 0.25,
                                              0, 7, 29.4]}``

    Examples:

        >>> e = ProlateEllipsoid(x=1, y=2, z=3, large_axis=6, small_axis=4,
        ...                      strike=10, dip=20, rake=30, props={
        ...                      'remanent magnetization': [10, 25, 40],
        ...                      'susceptibility tensor': [0.562, 0.485,
        ...                                                0.25, 90, 0,
        ...                                                0]})
        >>> e.props['remanent magnetization']
        [10, 25, 40]
        >>> e.addprop('density', 20)
        >>> print(e.props['density'])
        20
        >>> print(e)
        x:1 | y:2 | z:3 | large_axis:6 | small_axis:4 | strike:10 | dip:20 | r\
ake:30 | density:20 | remanent magnetization:[10, 25, 40] | susceptibility ten\
sor:[0.562, 0.485, 0.25, 90, 0, 0]
        >>> e = ProlateEllipsoid(1, 2, 3, 6, 4, 10, 20, 30)
        >>> print(e)
        x:1 | y:2 | z:3 | large_axis:6 | small_axis:4 | strike:10 | dip:20 | r\
ake:30
        >>> e.addprop('density', 2670)
        >>> print(e)
        x:1 | y:2 | z:3 | large_axis:6 | small_axis:4 | strike:10 | dip:20 | r\
ake:30 | density:2670

    """

    def __init__(self, x, y, z, large_axis, small_axis,
                 strike, dip, rake, props=None):
        super().__init__(props)

        self.x = x
        self.y = y
        self.z = z
        self.large_axis = large_axis
        self.small_axis = small_axis
        self.strike = strike
        self.dip = dip
        self.rake = rake

        assert self.large_axis > self.small_axis, "large_axis must be greater \
than small_axis"

        # Auxiliary orientation angles
        alpha, gamma, delta = _auxiliary_angles(self.strike,
                                                self.dip,
                                                self.rake)

        # Coordinate transformation matrix
        self.transf_matrix = _coord_transf_matrix_triaxial(alpha,
                                                           gamma,
                                                           delta)

    def __str__(self):
        """
        Return a string representation of the prolate ellipsoid.
        """

        names = [('x', self.x), ('y', self.y), ('z', self.z),
                 ('large_axis', self.large_axis),
                 ('small_axis', self.small_axis), ('strike', self.strike),
                 ('dip', self.dip), ('rake', self.rake)]
        names = names + [(p, self.props[p]) for p in sorted(self.props)]
        return ' | '.join('%s:%s' % (n, v) for n, v in names)

    @property
    def susceptibility_tensor(self):
        '''
        Calculate the susceptibility tensor (in SI) in the main system.

        The susceptibility tensor is calculated if
        'principal susceptibilities' and 'susceptibility angles' are
        defined in the dictionary of physical properties props.
        The 'principal susceptibilities' must be a list containing
        the three positive eigenvalues (principal susceptibilities k1, k2
        and k3) of the susceptibility tensor, in descending order. The
        'susceptibility angles' must be a list containing three angles
        (in degree) used to compute the eigenvector matrix U of the
        susceptibility tensor. The eigenvector matrix is defined by the
        function _coord_transf_matrix_triaxial.
        '''

        if 'principal susceptibilities' and 'susceptibility angles' \
                in self.props:

            assert len(self.props['principal susceptibilities']) == 3, \
                'there must be three principal susceptibilities'
            assert len(self.props['susceptibility angles']) == 3, \
                'there must be three angles'

            # Large, intermediate and small eigenvalues of the
            # susceptibility tensor (principal susceptibilities)
            k1 = self.props['principal susceptibilities'][0]
            k2 = self.props['principal susceptibilities'][1]
            k3 = self.props['principal susceptibilities'][2]

            assert k1 >= k2 >= k3, 'the principal susceptibilities must be \
given in descending order'

            assert (k1 > 0) and (k2 > 0) and (k3 > 0), 'the principal \
susceptibilities must be all positive'

            # Angles (in degrees) defining the eigenvector matrix
            # of the susceptibility tensor
            strike = self.props['susceptibility angles'][0]
            dip = self.props['susceptibility angles'][1]
            rake = self.props['susceptibility angles'][2]

            # Eigenvector matrix of the susceptibility tensor
            alpha, gamma, delta = _auxiliary_angles(strike, dip, rake)
            U = _coord_transf_matrix_triaxial(alpha, gamma, delta)

            suscep_tensor = numpy.dot(U, numpy.diag([k1, k2, k3]))
            suscep_tensor = numpy.dot(suscep_tensor, U.T)

            return suscep_tensor

        else:
            return None


class OblateEllipsoid(GeometricElement):
    """
    Create an arbitrarily-oriented oblate ellipsoid.

    Oblate ellipsoids are those having symmetry around the small axes.
    This code follows a convention similar to that defined by
    Emerson et al. (1985) and defines the spatial
    orientation of the ellipsoid by using three angles: strike,
    rake and dip. These angles are commonly used to define the
    orientation of geological structures (Allmendinger et al., 2012).

    References:

    Allmendinger, R., Cardozo, N., and Fisher, D. M.: Structural geology
    algorithms : vectors and tensors, Cambridge University Press, 2012.

    Emerson, D. W., Clark, D., and Saul, S.: Magnetic exploration models
    incorporating remanence, demagnetization and anisotropy: HP 41C
    handheld computer algorithms, Exploration Geophysics, 16, 1-122, 1985.

    .. note:: The coordinate system used is x -> North, y -> East
    and z -> Down

    Parameters:

    * x, y, z : float
        The coordinates of the center of the ellipsoid.
    * small_axis, large_axis: float
        Semi-axes forming the ellipsoid.
    * strike, dip, rake
        Orientation angles of the ellipsoid.
    * props : dict
        Physical properties assigned to the ellipsoid.
        Ex: ``props={'density':10,
                     'remanent magnetization':[10, 25, 40],
                     'susceptibility tensor':[0.562, 0.485, 0.25,
                                              90, 0, 0]}``

    Examples:

        >>> e = OblateEllipsoid(x=1, y=2, z=3, small_axis=4, large_axis=6,
        ...                     strike=10, dip=20, rake=30, props={
        ...                     'remanent magnetization': [10, 25, 40],
        ...                     'susceptibility tensor': [0.562, 0.485,
        ...                                               0.25, 90, 0,
        ...                                               0]})
        >>> e.props['remanent magnetization']
        [10, 25, 40]
        >>> e.addprop('density', 20)
        >>> print(e.props['density'])
        20
        >>> print(e)
        x:1 | y:2 | z:3 | small_axis:4 | large_axis:6 | strike:10 | dip:20 | r\
ake:30 | density:20 | remanent magnetization:[10, 25, 40] | susceptibility ten\
sor:[0.562, 0.485, 0.25, 90, 0, 0]
        >>> e = OblateEllipsoid(1, 2, 3, 2, 9, 10, 20, 30)
        >>> print(e)
        x:1 | y:2 | z:3 | small_axis:2 | large_axis:9 | strike:10 | dip:20 | r\
ake:30
        >>> e.addprop('density', 2670)
        >>> print(e)
        x:1 | y:2 | z:3 | small_axis:2 | large_axis:9 | strike:10 | dip:20 | r\
ake:30 | density:2670

    """

    def __init__(self, x, y, z, small_axis, large_axis,
                 strike, dip, rake, props=None):
        super().__init__(props)

        self.x = x
        self.y = y
        self.z = z
        self.small_axis = small_axis
        self.large_axis = large_axis
        self.strike = strike
        self.dip = dip
        self.rake = rake

        assert self.large_axis > self.small_axis, "large_axis must be greater \
than small_axis"

        # Auxiliary orientation angles
        alpha, gamma, delta = _auxiliary_angles(self.strike,
                                                self.dip,
                                                self.rake)

        # Coordinate transformation matrix
        self.transf_matrix = _coord_transf_matrix_triaxial(alpha,
                                                           gamma,
                                                           delta)

    def __str__(self):
        """
        Return a string representation of the oblate ellipsoid.
        """

        names = [('x', self.x), ('y', self.y), ('z', self.z),
                 ('small_axis', self.small_axis),
                 ('large_axis', self.large_axis), ('strike', self.strike),
                 ('dip', self.dip), ('rake', self.rake)]
        names = names + [(p, self.props[p]) for p in sorted(self.props)]
        return ' | '.join('%s:%s' % (n, v) for n, v in names)

    @property
    def susceptibility_tensor(self):
        '''
        Calculate the susceptibility tensor (in SI) in the main system.

        The susceptibility tensor is calculated if
        'principal susceptibilities' and 'susceptibility angles' are
        defined in the dictionary of physical properties props.
        The 'principal susceptibilities' must be a list containing
        the three positive eigenvalues (principal susceptibilities k1, k2
        and k3) of the susceptibility tensor, in descending order. The
        'susceptibility angles' must be a list containing three angles
        (in degree) used to compute the eigenvector matrix U of the
        susceptibility tensor. The eigenvector matrix is defined by the
        function _coord_transf_matrix_triaxial.
        '''

        if 'principal susceptibilities' and 'susceptibility angles' \
                in self.props:

            assert len(self.props['principal susceptibilities']) == 3, \
                'there must be three principal susceptibilities'
            assert len(self.props['susceptibility angles']) == 3, \
                'there must be three angles'

            # Large, intermediate and small eigenvalues of the
            # susceptibility tensor (principal susceptibilities)
            k1 = self.props['principal susceptibilities'][0]
            k2 = self.props['principal susceptibilities'][1]
            k3 = self.props['principal susceptibilities'][2]

            assert k1 >= k2 >= k3, 'the principal susceptibilities must be \
given in descending order'

            assert (k1 > 0) and (k2 > 0) and (k3 > 0), 'the principal \
susceptibilities must be all positive'

            # Angles (in degrees) defining the eigenvector matrix
            # of the susceptibility tensor
            strike = self.props['susceptibility angles'][0]
            dip = self.props['susceptibility angles'][1]
            rake = self.props['susceptibility angles'][2]

            # Eigenvector matrix of the susceptibility tensor
            alpha, gamma, delta = _auxiliary_angles(strike, dip, rake)
            U = _coord_transf_matrix_triaxial(alpha, gamma, delta)

            suscep_tensor = numpy.dot(U, numpy.diag([k1, k2, k3]))
            suscep_tensor = numpy.dot(suscep_tensor, U.T)

            return suscep_tensor

        else:
            return None


def _auxiliary_angles(strike, dip, rake):
    '''
    Calculate auxiliary angles alpha, gamma and delta (Clark et al., 1986)
    as functions of geological angles strike, dip and rake
    (Clark et al., 1986; Allmendinger et al., 2012), given in degrees.
    This function implements the formulas presented by
    Clark et al. (1986).

    References:

    Clark, D., Saul, S., and Emerson, D.: Magnetic and gravity anomalies
    of a triaxial ellipsoid, Exploration Geophysics, 17, 189-200, 1986.

    Allmendinger, R., Cardozo, N., and Fisher, D. M.:
    Structural geology algorithms : vectors and tensors,
    Cambridge University Press, 2012.
    '''

    strike_r = numpy.deg2rad(strike)
    cos_dip = numpy.cos(numpy.deg2rad(dip))
    sin_dip = numpy.sin(numpy.deg2rad(dip))
    cos_rake = numpy.cos(numpy.deg2rad(rake))
    sin_rake = numpy.sin(numpy.deg2rad(rake))

    aux = sin_dip*sin_rake
    aux1 = cos_rake/numpy.sqrt(1 - aux*aux)
    aux2 = sin_dip*cos_rake

    if aux1 > 1.:
        aux1 = 1.
    if aux1 < -1.:
        aux1 = -1.

    alpha = strike_r - numpy.arccos(aux1)
    if aux2 != 0:
        gamma = numpy.arctan(cos_dip/aux2)
    else:
        if cos_dip > 0:
            gamma = numpy.pi/2
        if cos_dip < 0:
            gamma = -numpy.pi/2
        if cos_dip == 0:
            gamma = 0
    delta = numpy.arcsin(aux)

    assert delta <= numpy.pi/2, 'delta must be lower than or equalt to 90 \
degrees'

    assert (gamma >= -numpy.pi/2) and (gamma <= numpy.pi/2), 'gamma must lie \
between -90 and 90 degrees.'

    return alpha, gamma, delta


def _coord_transf_matrix_triaxial(alpha, gamma, delta):
    '''
    Calculate the coordinate transformation matrix
    for triaxial or prolate ellipsoids by using the auxiliary angles
    alpha, gamma and delta.

    The columns of this matrix are defined according to the unit vectors
    v1, v2 and v3 presented by Clark et al. (1986, p. 192).

    References:

    Clark, D., Saul, S., and Emerson, D.: Magnetic and gravity anomalies
    of a triaxial ellipsoid, Exploration Geophysics, 17, 189-200, 1986.
    '''

    cos_alpha = numpy.cos(alpha)
    sin_alpha = numpy.sin(alpha)

    cos_gamma = numpy.cos(gamma)
    sin_gamma = numpy.sin(gamma)

    cos_delta = numpy.cos(delta)
    sin_delta = numpy.sin(delta)

    v1 = numpy.array([-cos_alpha*cos_delta, -sin_alpha*cos_delta,
                      -sin_delta])

    v2 = numpy.array([cos_alpha*cos_gamma*sin_delta +
                      sin_alpha*sin_gamma, sin_alpha*cos_gamma*sin_delta -
                      cos_alpha*sin_gamma, -cos_gamma*cos_delta])

    v3 = numpy.array([sin_alpha*cos_gamma - cos_alpha*sin_gamma*sin_delta,
                      -cos_alpha*cos_gamma -
                      sin_alpha*sin_gamma*sin_delta,
                      sin_gamma*cos_delta])

    transf_matrix = numpy.vstack((v1, v2, v3)).T

    return transf_matrix


def _coord_transf_matrix_oblate(alpha, gamma, delta):
    '''
    Calculate the coordinate transformation matrix
    for oblate ellipsoids by using the auxiliary angles
    alpha, gamma and delta.

    The columns of this matrix are defined by unit vectors
    v1, v2 and v3.
    '''

    cos_alpha = numpy.cos(alpha)
    sin_alpha = numpy.sin(alpha)

    cos_gamma = numpy.cos(gamma)
    sin_gamma = numpy.sin(gamma)

    cos_delta = numpy.cos(delta)
    sin_delta = numpy.sin(delta)

    v1 = numpy.array([-cos_alpha*sin_gamma*sin_delta +
                      sin_alpha*cos_gamma, -sin_alpha*sin_gamma*sin_delta -
                      cos_alpha*cos_gamma, sin_gamma*cos_delta])

    v2 = numpy.array([-cos_alpha*cos_delta, -sin_alpha*cos_delta,
                      -sin_delta])

    v3 = numpy.array([sin_alpha*sin_gamma + cos_alpha*cos_gamma*sin_delta,
                      -cos_alpha*sin_gamma +
                      sin_alpha*cos_gamma*sin_delta,
                      -cos_gamma*cos_delta])

    transf_matrix = numpy.vstack((v1, v2, v3)).T

    return transf_matrix
