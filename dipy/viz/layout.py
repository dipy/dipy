import numpy as np

from dipy.viz.utils import get_bounding_box_sizes
from dipy.viz.utils import get_grid_cells_position


class Layout(object):
    """ Provides functionalities for laying out actors in a 3D scene.
    """

    def apply(self, actors):
        """ Positions the actors according to a certain layout.
        """
        positions = self.compute_positions(actors)

        for a, pos in zip(actors, positions):
            anchor = np.array(getattr(a, 'anchor', (0, 0, 0)))
            a.AddPosition(pos - (np.array(a.GetCenter()) + anchor))

    def compute_positions(self, actors):
        """ Computes the 3D coordinates of some actors. """
        return []


class GridLayout(Layout):
    """ Provides functionalities for laying out actors in a 2D grid fashion.

    The `GridLayout` class lays the actors in a 2D structured grid aligned
    with the xy-plane.
    """
    def __init__(self, cell_padding=0, cell_shape="rect", aspect_ratio=16/9., dim=None):
        """
        Parameters
        ----------
        cell_padding : 2-tuple of float or float (optional)
            Each grid cell will be padded according to (pad_x, pad_y) i.e.
            horizontally and vertically. Padding is evenly distributed on each
            side of the cell. If a single float is provided then both pad_x and
            pad_y will have the same value.
        cell_shape : {'rect', 'square', 'diagonal'} (optional)
            Specifies the desired shape of every grid cell.
            'rect' ensures the cells are the tightest.
            'square' ensures the cells are as wide as high.
            'diagonal' ensures the content of the cells can be rotated without
                       colliding with content of the neighboring cells.
        aspect_ratio : float (optional)
            Aspect ratio of the grid (width/height). Default: 16:9.
        dim : tuple of int (optional)
            Dimension (nb_rows, nb_cols) of the grid. If provided,
            `aspect_ratio` will be ignored.
        """
        self.cell_shape = cell_shape
        self.aspect_ratio = aspect_ratio
        self.dim = dim
        self.cell_padding = (cell_padding, cell_padding) if type(cell_padding) is int else cell_padding

    def get_cells_shape(self, actors):
        """ Gets the 2D shape (on the xy-plane) of some actors according to
        `self.cell_shape`.

        Parameters
        ----------
        actors : list of `vtkProp3D` objects
            Actors from which to calculate the 2D shape.

        Returns
        -------
        list of 2-tuple
            The 2D shape (on the xy-plane) of every actors.
        """
        if self.cell_shape == "rect":
            bounding_box_sizes = np.asarray(list(map(get_bounding_box_sizes, actors)))
            cell_shape = np.max(bounding_box_sizes, axis=0)[:2]
            shapes = [cell_shape] * len(actors)
        elif self.cell_shape == "square":
            bounding_box_sizes = np.asarray(list(map(get_bounding_box_sizes, actors)))
            cell_shape = np.max(bounding_box_sizes, axis=0)[:2]
            shapes = [(max(cell_shape),)*2] * len(actors)
        elif self.cell_shape == "diagonal":
            # Size of every cell corresponds to the diagonal of the largest bounding box.
            longest_diagonal = np.max([a.GetLength() for a in actors])
            shapes = [(longest_diagonal, longest_diagonal)] * len(actors)
        else:
            raise ValueError("Unknown cell shape: '{0}'".format(self.cell_shape))

        return shapes

    def compute_positions(self, actors):
        """ Computes the 3D coordinates of some actors.

        The coordinates will lie on the xy-plane and form a 2D grid.

        Parameters
        ----------
        actors : list of `vtkProp3D` objects
            Actors to be layout in a grid manner.

        Returns
        -------
        list of 3-tuple
            The computed 3D coordinates of every actors.
        """
        shapes = self.get_cells_shape(actors)

        # Add padding, if any, around every cell.
        shapes = [np.array(self.cell_padding)/2. + s for s in shapes]
        positions = get_grid_cells_position(shapes, self.aspect_ratio, self.dim)
        return positions
