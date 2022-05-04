
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial.ckdtree import cKDTree

from dipy.tracking.streamline import set_number_of_points


class FastStreamlineSearch(object):
    def __init__(self, ref_streamlines, max_radius, nb_mpts=4, bin_size=20.0,
                 resampling=24, bidirectional=True):
        """ Fast Streamline Search (FFS)

        Generate the Binned K-D Tree structure with reference streamlines,
        using streamlines barycenter and mean-points.
        See [StOnge2021]_ for further details.

        Parameters
        ----------
        ref_streamlines : Streamlines
            Streamlines (ref) to generate the tree structure.
        max_radius : float
            The maximum radius for the search will be limited by this value,
            Used to compute bins overlap.
        nb_mpts : int
            Number of means points to improve computation speed.
        bin_size : float
            The maximum search radius will be limited by this value.
        bidirectional : bool
            Compute the smallest distance with and without flip.

        Notes
        -----
        Make sure that streamlines are aligned in the same space.

        References
        ----------
        .. [StOnge2021] St-Onge E. et al., Fast Tractography Streamline Search,
                        International Workshop on Computational Diffusion MRI,
                        pp. 82-95. Springer, Cham, 2021.
        """
        assert (resampling % nb_mpts == 0)
        assert (max_radius > 0)
        self.nb_mpts = nb_mpts
        self.bin_size = bin_size
        self.bidirectional = bidirectional
        self.resampling = resampling
        self.bin_overlap = max_radius

        # Resample streamlines
        self.fss_slines = self._resample(ref_streamlines)
        self.fss_nb_slines = len(self.fss_slines)

        if self.bidirectional:
            self.fss_slines = np.concatenate([self.fss_slines, np.flip(self.fss_slines, axis=1)])

        # Compute streamlines barycenter
        barycenters = self._slines_barycenters(self.fss_slines)

        # Compute bin shape (min, max, shape)
        self.min_box = np.min(barycenters, axis=0) - self.bin_overlap
        self.max_box = np.max(barycenters, axis=0) + self.bin_overlap

        self.bin_shape = ((self.max_box - self.min_box) // bin_size).astype(int) + 1

        # Compute the center of each bin
        all_bins = np.vstack(np.unravel_index(np.arange(np.prod(self.bin_shape)), self.bin_shape)).T
        bins_center = (all_bins*bin_size + self.min_box).astype(float) + bin_size/2.0

        # Assign a list of streamlines to each bin
        baryc_tree = cKDTree(barycenters)
        baryc_in_bin = baryc_tree.query_ball_point(bins_center, bin_size/2.0 + self.bin_overlap, p=np.inf)

        # Compute streamlines mean-points
        meanpts = self._slines_mean_points(self.fss_slines)

        # Compute bin indices, streamlines + mean-points tree
        self.bin_indices = np.flatnonzero(np.array(baryc_in_bin))
        self.bin_dict = {}
        for i in self.bin_indices:
            slines_id = np.asarray(baryc_in_bin[i])
            self.bin_dict[i] = (slines_id, cKDTree(meanpts[slines_id]))

    def radius_search(self, streamlines, radius, use_negative=True):
        """ Radius Search using Fast Streamline Search

        For each given streamlines, return all reference streamlines
        within the given radius. See [StOnge2021]_ for further details.

        Parameters
        ----------
        streamlines : Streamlines
            Streamlines to generate the tree structure.
        radius : float
            Search radius (with MDF / average L2 distance)
            must be smaller than max_radius when FFS was initialized.
        use_negative : bool
            When used with bidirectional,
            negative values are returned for reversed order neighbors.


        Returns
        -------
        res : scipy COOrdinates sparse matrix (nb_slines x nb_slines_ref)
            Adjacency matrix containing all neighbors within the given radius

        Notes
        -----
        Given streamlines should be already aligned with reference streamlines.

        References
        ----------
        .. [StOnge2021] St-Onge E. et al., Fast Tractography Streamline Search,
                        International Workshop on Computational Diffusion MRI,
                        pp. 82-95. Springer, Cham, 2021.
        """
        assert (radius <= self.bin_overlap)

        # Resample query streamlines
        q_slines = self._resample(streamlines)
        q_nb_slines = len(q_slines)

        # Compute streamlines barycenter
        q_baryc = self._slines_barycenters(q_slines)

        # Verify if each barycenter are inside the min max box
        u_bin, binned_slines_ids = self._barycenters_binning(q_baryc)

        # Adapting radius for L1 bounded query: sqrt(3) = 1.73205080756887729...
        l1_sum_dist = 1.73205081 * radius * self.nb_mpts

        # Search for all similar streamlines
        list_id = []
        list_id_ref = []
        list_dist = []
        for i in range(len(u_bin)):
            bin_id = u_bin[i]

            if bin_id in self.bin_dict:
                slines_id_ref, ref_tree = self.bin_dict[bin_id]
                slines_id = binned_slines_ids[i]

                mpts = self._slines_mean_points(q_slines[slines_id])

                # Compute Tree L1 Query with mean-points
                res = ref_tree.query_ball_point(mpts, l1_sum_dist, p=1, return_sorted=False)

                # Refine distance with the complete
                for s, ref_ids in enumerate(res):
                    cur_sline_id = slines_id[s]
                    ref_sline_ids = slines_id_ref[ref_ids]
                    d = mean_l2_func(q_slines[cur_sline_id], self.fss_slines[ref_sline_ids])
                    in_dist_max = d < radius

                    # Return all pairs within the radius
                    y = ref_sline_ids[in_dist_max]
                    x = np.full_like(y, cur_sline_id)
                    list_id.append(x)
                    list_id_ref.append(y)
                    list_dist.append(d[in_dist_max])

        # Combine all results in a coo sparse matrix
        if len(list_id) > 0:
            ids_in = np.hstack(list_id)
            ids_ref = np.hstack(list_id_ref)
            dist = np.hstack(list_dist)

            if self.bidirectional:
                flipped = ids_ref >= self.fss_nb_slines
                ids_ref[flipped] -= self.fss_nb_slines
                if use_negative:
                    dist[flipped] *= -1.0

            return coo_matrix((dist, (ids_in, ids_ref)),
                              shape=(q_nb_slines, self.fss_nb_slines))
        else:
            # No results, return an empty sparse matrix
            return coo_matrix(([], ([], [])),
                              shape=(q_nb_slines, self.fss_nb_slines))

    def _resample(self, streamlines):
        """Resample streamlines"""
        res_slines = np.zeros([len(streamlines), self.resampling, 3], dtype=np.float32)
        for i in range(len(streamlines)):
            if len(streamlines[i]) < 2:
                res_slines[i] = streamlines[i]
            else:
                res_slines[i] = set_number_of_points(streamlines[i], self.resampling)
        return res_slines

    def _slines_barycenters(self, slines_arr):
        """Compute streamlines barycenter"""
        return np.mean(slines_arr, axis=1)

    def _slines_mean_points(self, slines_arr):
        """Compute streamlines mean-points"""
        mpts = np.mean(slines_arr.reshape((len(slines_arr), self.nb_mpts, -1, 3)), axis=2)
        return mpts.reshape(len(slines_arr), -1)

    def _barycenters_binning(self, barycenters):
        """Bin indices in a list according to their barycenter position"""
        in_bin = np.logical_and(np.all(barycenters >= self.min_box, axis=1),
                                np.all(barycenters <= self.max_box, axis=1))

        baryc_bins = ((barycenters[in_bin] - self.min_box) // self.bin_size).astype(int)
        baryc_multiid = np.ravel_multi_index(baryc_bins.T, self.bin_shape)

        array_sortid = np.argsort(baryc_multiid)
        u_bin, mapping = np.unique(baryc_multiid[array_sortid], return_index=True)
        slines_ids = np.split(np.flatnonzero(in_bin)[array_sortid], mapping[1:])
        return u_bin, slines_ids


def nearest_from_matrix_row(coo_matrix):
    """
    Return the nearest (smallest) for each row given an coo sparse matrix

    Parameters
    ----------
    coo_matrix : scipy COOrdinates sparse matrix (nb_slines x nb_slines_ref)
        Adjacency matrix containing all neighbors within the given radius

    Returns
    -------
    non_zero_ids : numpy array (nb_non_empty_row x 1)
        Indices of each non-empty slines (row)
    nearest_id : numpy array (nb_non_empty_row x 1)
        Indices of the nearest reference match (column)
    nearest_dist : numpy array (nb_non_empty_row x 1)
        Distance for each nearest match
    """
    non_zero_ids = np.unique(coo_matrix.row)
    sparse_matrix = np.abs(coo_matrix.tocsr())
    upper_limit = np.max(sparse_matrix.data) + 1.0
    sparse_matrix.data = upper_limit - sparse_matrix.data
    nearest_id = np.squeeze(sparse_matrix.argmax(axis=1).data)[non_zero_ids]
    nearest_dist = upper_limit - np.squeeze(sparse_matrix.max(axis=1).data)
    return non_zero_ids, nearest_id, nearest_dist


def nearest_from_matrix_col(coo_matrix):
    """
    Return the nearest (smallest) for each column given an coo sparse matrix

    Parameters
    ----------
    coo_matrix : scipy COOrdinates sparse matrix (nb_slines x nb_slines_ref)
        Adjacency matrix containing all neighbors within the given radius

    Returns
    -------
    non_zero_ids : numpy array (nb_non_empty_col x 1)
        Indices of each non-empty reference (column)
    nearest_id : numpy array (nb_non_empty_col x 1)
        Indices of the nearest slines match (row)
    nearest_dist : numpy array (nb_non_empty_col x 1)
        Distance for each nearest match
    """
    non_zero_ids = np.unique(coo_matrix.col)
    sparse_matrix = np.abs(coo_matrix.tocsc())
    upper_limit = np.max(sparse_matrix.data) + 1.0
    sparse_matrix.data = upper_limit - sparse_matrix.data
    nearest_id = np.squeeze(sparse_matrix.argmax(axis=0).data)[non_zero_ids]
    nearest_dist = upper_limit - np.squeeze(sparse_matrix.max(axis=0).data)
    return non_zero_ids, nearest_id, nearest_dist


def mean_l2_func(a, b):
    """Streamlines distance function: Average L2 (MDF without flip)"""
    return np.mean(np.sqrt(np.sum((a - b) ** 2, axis=-1)), axis=-1)


def mean_l1_func(a, b):
    """Streamlines distance function: Average L1"""
    return np.mean(np.sum(np.abs(a - b), axis=-1), axis=-1)
