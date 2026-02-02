"""
FORCE Reconstruction Module

Dictionary-based matching for diffusion MRI microstructure estimation.
"""

import numpy as np


class IndexFlatIP:
    """
    Flat index for inner product similarity search.

    This is a pure Python/NumPy implementation providing
    FAISS-like interface for inner product search.

    Parameters
    ----------
    d : int
        Dimension of vectors.
    """

    def __init__(self, d):
        if d <= 0:
            raise ValueError(f"Dimension must be positive, got {d}")
        self.d = int(d)
        self.ntotal = 0
        self._xb = None

    def add(self, x):
        """
        Add vectors to the index.

        Parameters
        ----------
        x : ndarray (n, d)
            Vectors to add.
        """
        x = np.ascontiguousarray(x, dtype=np.float32)

        if x.ndim == 1:
            if len(x) != self.d:
                raise ValueError(
                    f"Vector dimension {len(x)} != index dimension {self.d}"
                )
            x = x.reshape(1, -1)

        if x.ndim != 2:
            raise ValueError(f"Expected 1D or 2D array, got {x.ndim}D")

        if x.shape[1] != self.d:
            raise ValueError(
                f"Vector dimension {x.shape[1]} != index dimension {self.d}"
            )

        if self._xb is None:
            self._xb = x.copy()
        else:
            self._xb = np.vstack([self._xb, x])

        self.ntotal = len(self._xb)

    def search(self, x, k):
        """
        Search for k nearest neighbors by inner product.

        Parameters
        ----------
        x : ndarray (n, d)
            Query vectors.
        k : int
            Number of neighbors.

        Returns
        -------
        distances : ndarray (n, k)
            Inner products (descending order).
        indices : ndarray (n, k)
            Neighbor indices.
        """
        if self.ntotal == 0:
            raise RuntimeError("Cannot search empty index")

        x = np.ascontiguousarray(x, dtype=np.float32)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.shape[1] != self.d:
            raise ValueError(
                f"Query dimension {x.shape[1]} != index dimension {self.d}"
            )

        k = min(k, self.ntotal)

        # Compute inner products
        scores = x @ self._xb.T

        # Get top-k indices
        indices = np.argsort(-scores, axis=1)[:, :k]
        distances = np.take_along_axis(scores, indices, axis=1)

        return distances.astype(np.float32), indices.astype(np.int64)

    def reset(self):
        """Remove all vectors from the index."""
        self._xb = None
        self.ntotal = 0
