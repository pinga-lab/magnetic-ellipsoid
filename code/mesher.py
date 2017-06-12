"""
Generate and operate the geometric elements representing ellipsoids
"""
from __future__ import division, absolute_import
from future.builtins import super
import numpy
import copy as cp


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

        # Coordinate transformation matrix
        self.transf_matrix = _coord_transf_matrix_triaxial(self.strike,
                                                           self.dip,
                                                           self.rake)

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
            strike_suscep = self.props['susceptibility angles'][0]
            dip_suscep = self.props['susceptibility angles'][1]
            rake_suscep = self.props['susceptibility angles'][2]

            # Eigenvector matrix of the susceptibility tensor
            U = _coord_transf_matrix_triaxial(strike_suscep,
                                              dip_suscep,
                                              rake_suscep)

            suscep_tensor = numpy.dot(U, numpy.diag([k1, k2, k3]))
            suscep_tensor = numpy.dot(suscep_tensor, U.T)

            return suscep_tensor

        else:
            return None

    @property
    def volume(self):
        '''
        Calculate the ellipsoid volume.
        '''

        cte = 4*numpy.pi/3  # constant
        volume = cte*self.large_axis*self.intermediate_axis*self.small_axis
        return volume


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

        # Coordinate transformation matrix
        self.transf_matrix = _coord_transf_matrix_triaxial(self.strike,
                                                           self.dip,
                                                           self.rake)

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
            strike_suscep = self.props['susceptibility angles'][0]
            dip_suscep = self.props['susceptibility angles'][1]
            rake_suscep = self.props['susceptibility angles'][2]

            # Eigenvector matrix of the susceptibility tensor
            U = _coord_transf_matrix_triaxial(strike_suscep,
                                              dip_suscep,
                                              rake_suscep)

            suscep_tensor = numpy.dot(U, numpy.diag([k1, k2, k3]))
            suscep_tensor = numpy.dot(suscep_tensor, U.T)

            return suscep_tensor

        else:
            return None

    @property
    def volume(self):
        '''
        Calculate the ellipsoid volume.
        '''

        cte = 4*numpy.pi/3  # constant
        volume = cte*self.large_axis*self.small_axis*self.small_axis
        return volume


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

        # Coordinate transformation matrix
        self.transf_matrix = _coord_transf_matrix_oblate(self.strike,
                                                         self.dip,
                                                         self.rake)

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
            strike_suscep = self.props['susceptibility angles'][0]
            dip_suscep = self.props['susceptibility angles'][1]
            rake_suscep = self.props['susceptibility angles'][2]

            # Eigenvector matrix of the susceptibility tensor
            U = _coord_transf_matrix_oblate(strike_suscep,
                                            dip_suscep,
                                            rake_suscep)

            suscep_tensor = numpy.dot(U, numpy.diag([k1, k2, k3]))
            suscep_tensor = numpy.dot(suscep_tensor, U.T)

            return suscep_tensor

        else:
            return None

    @property
    def volume(self):
        '''
        Calculate the ellipsoid volume.
        '''

        cte = 4*numpy.pi/3  # constant
        volume = cte*self.large_axis*self.large_axis*self.small_axis
        return volume


def _coord_transf_matrix_triaxial(strike, dip, rake):
    '''
    Calculate a coordinate transformation matrix for triaxial or prolate
    ellipsoids as a function of given geological angles strike, dip and rake
    (Allmendinger et al., 2012), given in degrees.

    The matrix obtained by performing succesive rotations around the
    axes forming the main-coordinate system.

    References:

    Allmendinger, R., Cardozo, N., and Fisher, D. M.: Structural geology
    algorithms : vectors and tensors, Cambridge University Press, 2012.

    '''

    # Transform the angles strike, dip and rake from degrees to radians
    strike_rad = numpy.deg2rad(strike)
    dip_rad = numpy.deg2rad(dip)
    rake_rad = numpy.deg2rad(rake)

    halfpi = numpy.pi/2

    A = _R1(halfpi)            # Rotation around x-axis
    B = _R2(strike_rad)        # Rotation around y-axis
    C = _R1(halfpi - dip_rad)  # Rotation around x-axis
    D = _R3(rake_rad)          # Rotation around z-axis

    # Resultant rotation for triaxial and prolate ellipsoids
    transf_matrix = _multi_dot([A, B, C, D])

    return transf_matrix


def _coord_transf_matrix_oblate(strike, dip, rake):
    '''
    Calculate a coordinate transformation matrix for oblate
    ellipsoids as a function of given geological angles strike, dip and rake
    (Allmendinger et al., 2012), given in degrees.

    The matrix obtained by performing succesive rotations around the
    axes forming the main-coordinate system.

    References:

    Allmendinger, R., Cardozo, N., and Fisher, D. M.: Structural geology
    algorithms : vectors and tensors, Cambridge University Press, 2012.

    '''

    # Transform the angles strike, dip and rake from degrees to radians
    strike_rad = numpy.deg2rad(strike)
    dip_rad = numpy.deg2rad(dip)
    rake_rad = numpy.deg2rad(rake)

    halfpi = numpy.pi/2
    pi = numpy.pi

    A = _R3(-halfpi)           # Rotation around z-axis
    B = _R1(pi)                # Rotation around x-axis
    C = _R3(strike_rad)        # Rotation around x-axis
    D = _R2(halfpi - dip_rad)  # Rotation around z-axis
    E = _R1(rake_rad)          # Rotation around x-axis

    # Resultant rotation for oblate ellipsoids
    transf_matrix = _multi_dot([A, B, C, D, E])

    return transf_matrix


def _multi_dot(matrices):
    '''
    Multiply an ordered list of matrices.

    Parameters:

    * matrices :  list of 2D numpy arrays
        List of matrices to be multiplied. The multiplication
        is performed by following the order of the elements of
        matrices.

    Returns:

    * resultant_matrix : 2D numpy array
        Resultant matrix obtained by multiplying the
        elements of matrices.
    '''

    shape = matrices[0].shape
    for mat in matrices[1:]:
        assert mat.shape == shape, 'All matrices must have the same shape'

    resultant_matrix = reduce(numpy.dot, matrices)

    return resultant_matrix


def _R1(angle):
    '''
    Orthogonal matrix performing a rotation around
    the x-axis of a Cartesian coordinate system.

    Parameters:
    * angle : float
        Rotation angle (in radians).

    Returns:
    * R : 2D numpy array
        Rotation matrix.
    '''

    cos_angle = numpy.cos(angle)
    sin_angle = numpy.sin(angle)

    R = numpy.array([[1, 0, 0],
                    [0, cos_angle, sin_angle],
                    [0, -sin_angle, cos_angle]])

    return R


def _R2(angle):
    '''
    Orthogonal matrix performing a rotation around
    the y-axis of a Cartesian coordinate system.

    Parameters:
    * angle : float
        Rotation angle (in radians).

    Returns:
    * R : 2D numpy array
        Rotation matrix.
    '''

    cos_angle = numpy.cos(angle)
    sin_angle = numpy.sin(angle)

    R = numpy.array([[cos_angle, 0, -sin_angle],
                     [0, 1, 0],
                     [sin_angle, 0, cos_angle]])

    return R


def _R3(angle):
    '''
    Orthogonal matrix performing a rotation around
    the z-axis of a Cartesian coordinate system.

    Parameters:
    * angle : float
        Rotation angle (in radians).

    Returns:
    * R : 2D numpy array
        Rotation matrix.
    '''

    cos_angle = numpy.cos(angle)
    sin_angle = numpy.sin(angle)

    R = numpy.array([[cos_angle, sin_angle, 0],
                     [-sin_angle, cos_angle, 0],
                     [0, 0, 1]])

    return R
