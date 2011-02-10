""" Factory function(s) for spheres """

from dipy.data import get_sphere

def sphere_vf_from(input):
    """ Return sphere vertices and faces from a variety of inputs

    Parameters
    ----------
    input : str or tuple or dict
        * str - a named sphere from dipy.data.get_sphere
        * tuple - the vertex, face tuple all ready to go
        * dict - with keys 'vertices', 'faces'

    Returns
    -------
    vertices : ndarray
        N,3 ndarray of sphere vertex coordinates
    faces : ndarray
        Indices into `vertices`
    """
    if hasattr(input, 'keys'):
        return input['vertices'], input['faces']
    if isinstance(input, basestring):
        return get_sphere(input)
    return input
