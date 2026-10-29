"""
========================================
Surface seeding for tractography
========================================

Surface seeding is a way to generate initial position for tractography
from cortical surfaces position :footcite:p:`StOnge2018`.
"""

from fury.io import read_mesh
from fury.utils import normals_from_v_f
import numpy as np

from dipy.data import get_fnames
from dipy.tracking.mesh import (
    random_coordinates_from_surface,
    seeds_from_surface_coordinates,
)
from dipy.viz import actor, window

###############################################################################
# Fetch and load a surface

brain_lh = get_fnames(name="fury_surface")
vertices, triangles, _ = read_mesh(str(brain_lh))

###############################################################################
# Display the surface
# ===============================================
#
# First, create a surface actor from the vertices and triangles

surface_actor = actor.surface(vertices, triangles)
scene = window.Scene()
scene.add(surface_actor)

show_manager = window.ShowManager(title="surface_seed1", size=(600, 600), scene=scene)
gizmo = actor.gizmo(show_manager)
actor.click_gizmo(show_manager, gizmo, "L")
show_manager.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Initial cortical surface
#
#
# Generate a list of seeding positions
# ==============================================================
#
# Choose the number of seed

nb_seeds = 100000
nb_triangles = len(triangles)

###############################################################################
# Get a list of triangles indices and trilinear coordinates for each seed

tri_idx, trilin_co = random_coordinates_from_surface(nb_triangles, nb_seeds)

###############################################################################
# Get the 3d cartesian position from triangles indices and trilinear
# coordinates

seed_pts = seeds_from_surface_coordinates(triangles, vertices, tri_idx, trilin_co)

###############################################################################
# Compute normal and get the normal direction for each seeds

normals = normals_from_v_f(vertices, triangles)
seed_n = seeds_from_surface_coordinates(triangles, normals, tri_idx, trilin_co)

###############################################################################
# Create point actor for seeds (blue)

seed_actor = actor.point(seed_pts, colors=(0, 0, 1), size=6.0)

###############################################################################
# Create streamline actors for seeds normals (green outside, red inside)

normal_length = 0.5
normal_in = np.tile(seed_pts[:, np.newaxis, :], (1, 2, 1))
normal_out = np.tile(seed_pts[:, np.newaxis, :], (1, 2, 1))
normal_in[:, 0] -= seed_n * normal_length
normal_out[:, 1] += seed_n * normal_length

normal_in_actor = actor.streamlines(
    normal_in, colors=(1, 0, 0), thickness=0.05, outline_thickness=0
)
normal_out_actor = actor.streamlines(
    normal_out, colors=(0, 1, 0), thickness=0.05, outline_thickness=0
)

###############################################################################
# Visualise seeds and normals along the surface. The orientation gizmo in the
# bottom-left corner snaps the camera to the side of the clicked label. Here
# the click on the ``L`` label is simulated to get a left lateral view.

scene.add(seed_actor, normal_in_actor, normal_out_actor)
show_manager = window.ShowManager(title="surface_seed2", size=(600, 600), scene=scene)
gizmo = actor.gizmo(show_manager)
actor.click_gizmo(show_manager, gizmo, "L")
show_manager.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Surface seeds with normal orientation
#
#
# References
# ----------
#
# .. footbibliography::
#
