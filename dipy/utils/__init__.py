# code support utilities for dipy


def str2bool(txt):
    """ Convert string to a boolean value.

    References
    ----------
    https://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python/715468#715468
    """
    return str(txt).lower() in ("yes", "true", "t", "1")
