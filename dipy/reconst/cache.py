from dipy.core.onetime import auto_attr


class Cache:
    """Cache values based on a key object (such as a sphere or gradient table).

    Notes
    -----
    This class is meant to be used as a mix-in::

        class MyModel(Model, Cache):
            pass

        class MyModelFit(Fit):
            pass

    Inside a method on the fit, typical usage would be::

        def odf(sphere):
            M = self.model.cache_get('odf_basis_matrix', key=sphere)

            if M is None:
                M = self._compute_basis_matrix(sphere)
                self.model.cache_set('odf_basis_matrix', key=sphere, value=M)

    """

    # We use this method instead of __init__ to construct the cache, so
    # that the class can be used as a mixin, without having to worry about
    # calling the super-class constructor
    @auto_attr
    def _cache(self):
        return {}

    def cache_set(self, tag, key, value):
        """Store a value in the cache.

        Parameters
        ----------
        tag : str
            Description of the cached value.
        key : object
            Key object used to look up the cached value.
        value : object
            Value stored in the cache for each unique combination
            of ``(tag, key)``.

        Examples
        --------
        >>> def compute_expensive_matrix(parameters):
        ...     # Imagine the following computation is very expensive
        ...     return (p**2 for p in parameters)

        >>> c = Cache()

        >>> parameters = (1, 2, 3)
        >>> X1 = compute_expensive_matrix(parameters)

        >>> c.cache_set('expensive_matrix', parameters, X1)
        >>> X2 = c.cache_get('expensive_matrix', parameters)

        >>> X1 is X2
        True

        """
        self._cache[(tag, key)] = value

    def cache_get(self, tag, key, default=None):
        """Retrieve a value from the cache.

        Parameters
        ----------
        tag : str
            Description of the cached value.
        key : object
            Key object used to look up the cached value.
        default : object
            Value to be returned if no cached entry is found.

        Returns
        -------
        v : object
            Value from the cache associated with ``(tag, key)``.  Returns
            `default` if no cached entry is found.

        """
        return self._cache.get((tag, key), default)

    def cache_clear(self):
        """Clear the cache.

        """
        self._cache = {}
