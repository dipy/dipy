from collections import OrderedDict
from copy import deepcopy
from itertools import product
import logging

from nibabel.affines import apply_affine
import numpy as np

from dipy.io.utils import (
    Origin,
    Space,
    get_reference_info,
    is_header_compatible,
    is_reference_info_valid,
)
from dipy.io.vtk import convert_to_polydata

logger = logging.getLogger("StatefulSurface")
logger.setLevel(level=logging.INFO)


def set_sfs_logger_level(log_level):
    """Change the logger of the StatefulSurface
    to one on the following: DEBUG, INFO, WARNING, CRITICAL, ERROR

    Parameters
    ----------
    log_level : str
        Log level for the StatefulSurface only
    """
    logger.setLevel(level=log_level)


class StatefulSurface:
    """Class for stateful representation of meshes and lines
    Object designed to be identical no matter the file format
    (gii, vtk, ply, stl, obj, pial). Facilitate transformation between space
    and data manipulation for each streamline / vertex.
    """

    def __init__(
        self,
        vertices,
        faces,
        reference,
        space,
        *,
        origin=Origin.NIFTI,
        data_per_vertex=None,
        dtype_dict=None,
    ):
        """Create a strict, state-aware, robust surface

        Parameters
        ----------
        vertices : ndarray
            Vertices of the surface, shape (N, 3) where N is the number of
            points on the surface.
        faces : ndarray
            Faces of the surface, shape (M, 3) where M is the number of
            triangles on the surface. Each face is defined by 3 indices
            corresponding to the vertices.
        reference : Nifti or Trk filename, Nifti1Image or TrkFile,
            Nifti1Header, trk.header (dict) or another Stateful Surface
            Reference that provides the spatial attributes.
            Typically a nifti-related object from the native space used for
            surface generation
        space : Enum (dipy.io.utils.Space)
            Current space in which the surface are (vox, voxmm or rasmm)
            After tracking the space is VOX, after loading with nibabel
            the space is RASMM
        origin : Enum (dipy.io.utils.Origin), optional
            Current origin in which the surface are (center or corner)
            After loading with nibabel the origin is CENTER
        data_per_vertex : dict, optional
            Dictionary in which each key has X items.
            X being the number of points on the surface.
        dtype_dict : dict, optional
            Dictionary containing the desired datatype for vertices, faces
            and all data_per_vertex keys.

        Notes
        -----
        Very important to respect the convention, verify that surface
        match the reference and are effectively in the right space.

        Any change to the number of surface's vertices, data_per_vertex
        requires verification.

        In a case of manipulation not allowed by this object, use Nibabel
        directly and be careful.
        """

        self.fs_metadata = None
        self.gii_header = None

        self.data_per_vertex = {} if data_per_vertex is None else data_per_vertex
        if dtype_dict is None:
            dtype_dict = {"vertices": np.float64, "faces": np.uint32}
        self.dtype_dict = dtype_dict

        self._vertices = np.array(vertices, dtype=self.dtype_dict["vertices"])
        self._faces = np.array(faces, dtype=self.dtype_dict["faces"])

        # Verify that vertices and faces are in the right format (if not empty)
        if self._vertices.size != 0 or self._faces.size != 0:
            if self._vertices.ndim != 2 or self._vertices.shape[1] != 3:
                raise ValueError(
                    "Vertices must be a 2D array with shape (N, 3), "
                    f"got {self._vertices.shape} instead."
                )
            if self._faces.ndim != 2 or self._faces.shape[1] != 3:
                raise ValueError(
                    "Faces must be a 2D array with shape (M, 3), "
                    f"got {self._faces.shape} instead."
                )

        if isinstance(reference, type(self)):
            logger.warning(
                "Using a StatefulSurface as reference, this "
                "will copy only the space_attributes, not "
                "the state. The variables space and origin "
                "must be specified separately."
            )
            logger.warning(
                "To copy the state from another StatefulSurface "
                "you may want to use the function from_sfs "
                "(static function of the StatefulSurface)."
            )

        if isinstance(reference, tuple) and len(reference) == 4:
            if is_reference_info_valid(*reference):
                space_attributes = reference
            else:
                raise TypeError(
                    "The provided space attributes are not "
                    "considered valid, please correct before "
                    "using them with StatefulSurface."
                )
        else:
            space_attributes = get_reference_info(reference)
            if space_attributes is None:
                raise TypeError(
                    "Reference MUST be one of the following:\n"
                    "Nifti or Trk filename, Nifti1Image or "
                    "TrkFile, Nifti1Header or trk.header (dict)."
                )

        self._affine, self._dimensions, self._voxel_sizes, self._voxel_order = (
            space_attributes
        )
        self._inv_affine = np.linalg.inv(self._affine).astype(np.float64)

        if space not in Space:
            raise ValueError("Space MUST be from Space enum, e.g Space.VOX.")
        self._space = space

        if origin not in Origin:
            raise ValueError("Origin MUST be from Origin enum, e.g Origin.NIFTI.")
        self._origin = origin

        logger.debug(self)

    @staticmethod
    def are_compatible(sfs_1, sfs_2):
        """Compatibility verification of two StatefulSurface to ensure space,
        origin, data_per_vertex consistency"""

        are_sfs_compatible = True
        if not is_header_compatible(sfs_1, sfs_2):
            logger.warning("Inconsistent spatial attributes between both sfs.")
            are_sfs_compatible = False

        if sfs_1.space != sfs_2.space:
            logger.warning("Inconsistent space between both sfs.")
            are_sfs_compatible = False
        if sfs_1.origin != sfs_2.origin:
            logger.warning("Inconsistent origin between both sfs.")
            are_sfs_compatible = False

        if sfs_1.get_data_per_vertex_keys() != sfs_2.get_data_per_vertex_keys():
            logger.warning("Inconsistent data_per_vertex between both sfs.")
            are_sfs_compatible = False

        return are_sfs_compatible

    @staticmethod
    def from_sfs(vertices, sfs, *, faces=None, data_per_vertex=None):
        """Create an instance of `StatefulSurface` from another instance
        of `StatefulSurface`.

        Parameters
        ----------
        vertices : ndarray
            Vertices of the new StatefulSurface.
        sfs : StatefulSurface,
            The other StatefulSurface to copy the space_attribute AND
            state from.
        faces : ndarray, optional
            Faces of the new StatefulSurface. If None, the faces of the
            original StatefulSurface will be used.
        data_per_vertex : dict, optional
            Dictionary in which each key has X items.
            X being the number of points on the surface.
        -----
        """
        faces = faces if faces is not None else sfs.faces
        new_sfs = StatefulSurface(
            vertices,
            faces,
            sfs.space_attributes,
            sfs.space,
            origin=sfs.origin,
            data_per_vertex=data_per_vertex,
        )
        new_sfs.dtype_dict = sfs.dtype_dict
        return new_sfs

    def __str__(self):
        """Generate the string for printing"""
        affine = np.array2string(
            self._affine, formatter={"float_kind": lambda x: f"{x:.6f}"}
        )
        vox_sizes = np.array2string(
            self._voxel_sizes, formatter={"float_kind": lambda x: f"{x:.2f}"}
        )

        text = f"Affine: \n{affine}"
        text += f"\ndimensions: {np.array2string(self._dimensions)}"
        text += f"\nvoxel_sizes: {vox_sizes}"
        text += f"\nvoxel_order: {self._voxel_order}"

        text += f"\nface_count: {self._get_face_count()}"
        text += f"\nvertex_count: {self._get_vertex_count()}"
        text += f"\ndata_per_vertex keys: {self.get_data_per_vertex_keys()}"

        return text

    def __len__(self):
        """Define the length of the object"""
        return self._get_vertex_count()

    def __getitem__(self, key):
        # TODO: implement slicing if make sense
        pass

    def __eq__(self, other):
        """Robust StatefulSurface equality test"""
        if not self.are_compatible(self, other):
            return False

        vertices_equal = np.allclose(self._vertices, other.vertices, rtol=1e-3)
        faces_equal = np.allclose(self._faces, other.faces, rtol=1e-3)

        if not vertices_equal or not faces_equal:
            return False

        dpp_equal = True
        for key in self.data_per_vertex:
            dpp_equal = dpp_equal and np.allclose(
                self.data_per_vertex[key].get_data(),
                other.data_per_vertex[key].get_data(),
                rtol=1e-3,
            )
        if not dpp_equal:
            return False

        return True

    def __ne__(self, other):
        """Robust StatefulSurface equality test (NOT)"""
        return not self == other

    def __add__(self, other_sfs):
        """Addition of two sfs with attributes consistency checks"""
        # TODO
        pass

    def __iadd__(self, other):
        # TODO
        pass

    @property
    def dtype_dict(self):
        """Getter for dtype_dict"""

        if not hasattr(self, "_vertices") or not hasattr(self, "_faces"):
            dtype_dict = {"vertices": np.float64, "faces": np.uint32}
            return OrderedDict(dtype_dict)

        dtype_dict = {
            "vertices": self._vertices.dtype,
            "faces": self._faces.dtype,
        }
        if self.data_per_vertex is not None:
            dtype_dict["dpp"] = {}
            for key in self.data_per_vertex.keys():
                if key in self.data_per_vertex:
                    dtype_dict["dpp"][key] = self.data_per_vertex[key].dtype

        return OrderedDict(dtype_dict)

    @property
    def space_attributes(self):
        """Getter for spatial attribute"""
        return self._affine, self._dimensions, self._voxel_sizes, self._voxel_order

    @property
    def space(self):
        """Getter for the current space"""
        return self._space

    @property
    def affine(self):
        """Getter for the reference affine"""
        return self._affine

    @property
    def dimensions(self):
        """Getter for the reference dimensions"""
        return self._dimensions

    @property
    def voxel_sizes(self):
        """Getter for the reference voxel sizes"""
        return self._voxel_sizes

    @property
    def voxel_order(self):
        """Getter for the reference voxel order"""
        return self._voxel_order

    @property
    def origin(self):
        """Getter for origin standard"""
        return self._origin

    @property
    def vertices(self):
        """Partially safe getter for vertices"""
        return self._vertices

    @property
    def faces(self):
        """Partially safe getter for faces"""
        return self._faces

    @dtype_dict.setter
    def dtype_dict(self, dtype_dict):
        """Modify dtype_dict.

        Parameters
        ----------
        dtype_dict : dict
            Dictionary containing the desired datatype for positions, offsets
            and all dpp and dps keys. (To use with TRX file format):
        """
        if not hasattr(self, "_vertices") or not hasattr(self, "_faces"):
            self.dtype_dict[0] = 0
            self.dtype_dict["vertices"] = dtype_dict["vertices"]
            self.dtype_dict["faces"] = dtype_dict["faces"]
            return

        if "faces" in dtype_dict:
            self._faces = self._faces.astype(dtype_dict["faces"])
        if "vertices" in dtype_dict:
            self._vertices = self._vertices.astype(dtype_dict["vertices"])

        if "dpp" not in dtype_dict:
            dtype_dict["dpp"] = {}

        for key in self.data_per_vertex:
            if key in dtype_dict["dpp"]:
                dtype_to_use = dtype_dict["dpp"][key]
                self.data_per_vertex[key] = self.data_per_vertex[key].astype(
                    dtype_to_use
                )

    def get_vertices_copy(self):
        """Safe getter for vertices (for slicing)"""
        return self._vertices.copy()

    def get_polydata(self):
        return convert_to_polydata(self._vertices, self._faces, self._data_per_vertex)

    @vertices.setter
    def vertices(self, data):
        """Modify surface. Creating a new object would be less risky.
        Parameters
        ----------
        data
        """
        if len(data) != len(self._vertices):
            raise ValueError("Number of vertices does not match.")
        self._vertices = data

    @property
    def data_per_vertex(self):
        """Getter for data_per_vertex"""
        return self._data_per_vertex

    @data_per_vertex.setter
    def data_per_vertex(self, data):
        """Modify vertex data . Creating a new object would be less risky.

        Parameters
        ----------
        data : dict
            Dictionary in which each key has X items.
            X being the number of points on the surface.
        """
        self._data_per_vertex = data
        logger.warning("data_per_vertex has been modified.")

    def get_data_per_vertex_keys(self):
        """Return a list of the data_per_vertex attribute names"""
        return list(set(self.data_per_vertex.keys()))

    def to_vox(self):
        """Safe function to transform vertices and update state"""
        if self._space == Space.VOXMM:
            self._voxmm_to_vox()
        elif self._space == Space.RASMM:
            self._rasmm_to_vox()
        elif self._space == Space.LPSMM:
            self._lpsmm_to_vox()

    def to_voxmm(self):
        """Safe function to transform vertices and update state"""
        if self._space == Space.VOX:
            self._vox_to_voxmm()
        elif self._space == Space.RASMM:
            self._rasmm_to_voxmm()
        elif self._space == Space.LPSMM:
            self._lpsmm_to_voxmm()

    def to_rasmm(self):
        """Safe function to transform vertices and update state"""
        if self._space == Space.VOX:
            self._vox_to_rasmm()
        elif self._space == Space.VOXMM:
            self._voxmm_to_rasmm()
        elif self._space == Space.LPSMM:
            self._lpsmm_to_rasmm()

    def to_lpsmm(self):
        if self._space == Space.VOX:
            self._vox_to_lpsmm()
        elif self._space == Space.VOXMM:
            self._voxmm_to_lpsmm()
        elif self._space == Space.RASMM:
            self._rasmm_to_lpsmm()

    def to_space(self, target_space):
        """Safe function to transform vertices to a particular space using
        an enum and update state"""
        if target_space == Space.VOX:
            self.to_vox()
        elif target_space == Space.VOXMM:
            self.to_voxmm()
        elif target_space == Space.RASMM:
            self.to_rasmm()
        elif target_space == Space.LPSMM:
            self.to_lpsmm()
        else:
            logger.error(
                "Unsupported target space, please use Enum in dipy.io.stateful_surface."
            )

    def to_origin(self, target_origin):
        """Safe function to change vertices to a particular origin standard
        False means NIFTI (center) and True means TrackVis (corner)"""
        if target_origin == Origin.NIFTI:
            self.to_center()
        elif target_origin == Origin.TRACKVIS:
            self.to_corner()
        else:
            logger.error(
                "Unsupported origin standard, please use Enum in "
                "dipy.io.stateful_surface."
            )

    def to_center(self):
        """Safe function to shift vertices so the center of voxel is
        the origin"""
        if self._origin == Origin.TRACKVIS:
            self._shift_voxel_origin()

    def to_corner(self):
        """Safe function to shift vertices so the corner of voxel is
        the origin"""
        if self._origin == Origin.NIFTI:
            self._shift_voxel_origin()

    def compute_bounding_box(self):
        """Compute the bounding box of the vertices in their current state

        Returns
        -------
        output : ndarray
            8 corners of the XYZ aligned box, all zeros if no vertices
        """
        if self._vertices.size > 0:
            bbox_min = np.min(self._vertices, axis=0)
            bbox_max = np.max(self._vertices, axis=0)
            return np.asarray(list(product(*zip(bbox_min, bbox_max))))

        return np.zeros((8, 3))

    def is_bbox_in_vox_valid(self):
        """Verify that the bounding box is valid in voxel space.
        Negative coordinates or coordinates above the volume dimensions
        are considered invalid in voxel space.

        Returns
        -------
        output : bool
            Are the vertices within the volume of the associated reference
        """
        if not self._vertices.size:
            return True

        old_space = deepcopy(self.space)
        old_origin = deepcopy(self.origin)

        # Do to rotation, equivalent of a OBB must be done
        self.to_vox()
        self.to_corner()
        bbox_corners = deepcopy(self.compute_bounding_box())

        is_valid = True
        if np.any(bbox_corners < -1e-3):
            logger.error("Voxel space values lower than 0.0.")
            logger.debug(bbox_corners)
            is_valid = False

        if (
            np.any(bbox_corners[:, 0] > self._dimensions[0])
            or np.any(bbox_corners[:, 1] > self._dimensions[1])
            or np.any(bbox_corners[:, 2] > self._dimensions[2])
        ):
            logger.error("Voxel space values higher than dimensions.")
            logger.debug(bbox_corners)
            is_valid = False

        self.to_space(old_space)
        self.to_origin(old_origin)

        return is_valid

    def remove_invalid_vertices(self, *, epsilon=1e-3):
        # TODO: implement if make sense (must cut faces too)
        raise NotImplementedError

    def _get_face_count(self):
        """Safe getter for the number of faces"""
        return len(self._faces)

    def _get_vertex_count(self):
        """Safe getter for the number of vertices"""
        return len(self._vertices)

    def _vox_to_voxmm(self):
        """Unsafe function to transform vertices"""
        if self._space == Space.VOX:
            if self._vertices.size > 0:
                self._vertices *= np.asarray(self._voxel_sizes)
            self._space = Space.VOXMM
            logger.debug("Moved vertices from vox to voxmm.")
        else:
            logger.warning("Wrong initial space for this function.")

    def _voxmm_to_vox(self):
        """Unsafe function to transform vertices"""
        if self._space == Space.VOXMM:
            if self._vertices.size > 0:
                self._vertices /= np.asarray(self._voxel_sizes)
            self._space = Space.VOX
            logger.debug("Moved vertices from voxmm to vox.")
        else:
            logger.warning("Wrong initial space for this function.")

    def _vox_to_rasmm(self):
        """Unsafe function to transform vertices"""
        if self._space == Space.VOX:
            if self._vertices.size > 0:
                self._vertices = apply_affine(
                    self._affine, self._vertices, inplace=True
                )
            self._space = Space.RASMM
            logger.debug("Moved vertices from vox to rasmm.")
        else:
            logger.warning("Wrong initial space for this function.")

    def _rasmm_to_vox(self):
        """Unsafe function to transform vertices"""
        if self._space == Space.RASMM:
            if self._vertices.size > 0:
                self._vertices = apply_affine(
                    self._inv_affine, self._vertices, inplace=True
                )
            self._space = Space.VOX
            logger.debug("Moved vertices from rasmm to vox.")
        else:
            logger.warning("Wrong initial space for this function.")

    def _voxmm_to_rasmm(self):
        """Unsafe function to transform vertices"""
        if self._space == Space.VOXMM:
            if self._vertices.size > 0:
                self._vertices /= np.asarray(self._voxel_sizes)
                self._vertices = apply_affine(
                    self._affine, self._vertices, inplace=True
                )
            self._space = Space.RASMM
            logger.debug("Moved vertices from voxmm to rasmm.")
        else:
            logger.warning("Wrong initial space for this function.")

    def _rasmm_to_voxmm(self):
        """Unsafe function to transform vertices"""
        if self._space == Space.RASMM:
            if self._vertices.size > 0:
                self._vertices = apply_affine(
                    self._inv_affine, self._vertices, inplace=True
                )
                self._vertices *= np.asarray(self._voxel_sizes)
            self._space = Space.VOXMM
            logger.debug("Moved vertices from rasmm to voxmm.")
        else:
            logger.warning("Wrong initial space for this function.")

    def _lpsmm_to_rasmm(self):
        """Unsafe function to transform vertices"""
        if self._space == Space.LPSMM:
            if self._vertices.size > 0:
                flip_affine = np.diag([-1, -1, 1, 1])
                self._vertices = apply_affine(flip_affine, self._vertices, inplace=True)
            self._space = Space.RASMM
            logger.debug("Moved vertices from lpsmm to rasmm.")
        else:
            logger.warning("Wrong initial space for this function.")

    def _rasmm_to_lpsmm(self):
        """Unsafe function to transform vertices"""
        if self._space == Space.RASMM:
            if self._vertices.size > 0:
                flip_affine = np.diag([-1, -1, 1, 1])
                self._vertices = apply_affine(flip_affine, self._vertices, inplace=True)
            self._space = Space.LPSMM
            logger.debug("Moved vertices from lpsmm to rasmm.")
        else:
            logger.warning("Wrong initial space for this function.")

    def _lpsmm_to_voxmm(self):
        """Unsafe function to transform vertices"""
        if self._space == Space.LPSMM:
            self._lpsmm_to_rasmm()
            self._rasmm_to_voxmm()
            logger.debug("Moved vertices from lpsmm to voxmm.")
        else:
            logger.warning("Wrong initial space for this function.")

    def _voxmm_to_lpsmm(self):
        """Unsafe function to transform vertices"""
        if self._space == Space.VOXMM:
            self._voxmm_to_rasmm()
            self._rasmm_to_lpsmm()
            logger.debug("Moved vertices from voxmm to lpsmm.")
        else:
            logger.warning("Wrong initial space for this function.")

    def _lpsmm_to_vox(self):
        """Unsafe function to transform vertices"""
        if self._space == Space.LPSMM:
            self._lpsmm_to_rasmm()
            self._rasmm_to_vox()
            logger.debug("Moved vertices from lpsmm to vox.")
        else:
            logger.warning("Wrong initial space for this function.")

    def _vox_to_lpsmm(self):
        """Unsafe function to transform vertices"""
        if self._space == Space.VOX:
            self._vox_to_rasmm()
            self._rasmm_to_lpsmm()
            logger.debug("Moved vertices from vox to lpsmm.")
        else:
            logger.warning("Wrong initial space for this function.")

    def _shift_voxel_origin(self):
        """Unsafe function to switch the origin from center to corner
        and vice versa"""
        if self._vertices.size:
            shift = np.asarray([0.5, 0.5, 0.5])
            if self._space == Space.VOXMM:
                shift = shift * self._voxel_sizes
            elif self._space == Space.RASMM or self._space == Space.LPSMM:
                tmp_affine = np.eye(4)
                tmp_affine[0:3, 0:3] = self._affine[0:3, 0:3]
                flip = [1, 1, 1] if self._space == Space.RASMM else [-1, -1, 1]
                shift = apply_affine(tmp_affine, shift) * flip
            if self._origin == Origin.TRACKVIS:
                shift *= -1

            self._vertices += shift

        if self._origin == Origin.NIFTI:
            logger.debug("Origin moved to the corner of voxel.")
            self._origin = Origin.TRACKVIS
        else:
            logger.debug("Origin moved to the center of voxel.")
            self._origin = Origin.NIFTI
