"""
=========================================
Introduction to interactive visualization
=========================================

In DIPY we created a thin interface to access many of the capabilities
available in the Visualization Toolkit framework (VTK) but tailored to the
needs of structural and diffusion imaging. Initially the 3D visualization
module was named ``fvtk``, meaning functions using vtk. This is still available
for backwards compatibility but now there is a more comprehensive way to access
the main functions using the following import.
"""

from dipy.viz import window, actor, widgets

"""
The main objects/functions which are used for drawing actors (e.g. slices,
streamlines) in a window or in a file are available in window. And the actors
are available in actor. There are also some objects which allow to add buttons
and slider and these interact both with windows and actors and those are in
widjets.
"""










