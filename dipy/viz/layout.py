import itertools
import numpy as np

from dipy.viz.utils import get_bounding_box_sizes
from dipy.viz.utils import get_grid_cells_position


class Layout(object):

    def apply(self, actors, anchors=[]):
        positions = self.compute_positions(actors)
        for a, pos in zip(actors, positions):
            anchor = a.GetCenter()
            if hasattr(a, "anchor") and a.anchor is not None:
                anchor = a.anchor

            a.SetPosition(pos - anchor)

    def compute_positions(self, actors):
        return []


class ColumnLayout(Layout):
    def __init__(self, padding=0):
        #self.axis = axis
        self.padding = (padding, padding) if type(padding) is int else padding

    def get_cells_shape(self, actors):
        bounding_box_sizes = np.asarray(list(map(get_bounding_box_sizes, actors)))
        max_width = np.max(bounding_box_sizes[:, 0])
        shapes = np.asarray([(max_width, height) for height in bounding_box_sizes[:, 1]])
        return shapes

    def compute_positions(self, actors):
        shapes = self.get_cells_shape(actors)

        X = np.zeros(len(actors))
        Y = np.cumsum(shapes[:, 1]) - shapes[:, 1]/2.
        Y -= Y[0]  # Start at (0, 0)
        Z = np.zeros(len(actors))

        positions = np.array([X, -Y, Z]).T
        return positions

    def get_borders(self, actors, color=(1, 0, 0), linewidth=1):
        from dipy.viz import actor

        shapes = self.get_cells_shape(actors)
        positions = self.compute_positions(actors)

        print positions
        borders = []
        for pos, shape in zip(positions, shapes):
            offset = np.array(shape)/2.
            border = np.array([pos + (offset[0], offset[1], 0),
                               pos + (-offset[0], offset[1], 0),
                               pos + (-offset[0], -offset[1], 0),
                               pos + (offset[0], -offset[1], 0),
                               pos + (offset[0], offset[1], 0)])

            borders.append(border)

        return actor.line(borders, colors=color, linewidth=linewidth)


class GridLayout(Layout):
    def __init__(self, padding=0, cell_shape="rect", aspect_ratio=16/9., dim=None):
        self.cell_shape = cell_shape
        self.aspect_ratio = aspect_ratio
        self.dim = dim
        self.padding = (padding, padding) if type(padding) is int else padding

    def get_cells_shape(self, actors):
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
        shapes = self.get_cells_shape(actors)

        # Add margin around every cell.
        shapes = [np.array(self.padding)/2. + s for s in shapes]
        positions = get_grid_cells_position(shapes, self.aspect_ratio, self.dim)
        return positions

    def get_borders(self, actors, color=(1, 0, 0), linewidth=1):
        from dipy.viz import actor

        shapes = self.get_cells_shape(actors)
        positions = self.compute_positions(actors)

        #print positions
        borders = []
        for pos, shape, a in zip(positions, shapes, actors):
            offset = np.array(shape)/2.
            #pos = np.array(a.GetPosition())
            #anchor = a.GetCenter()
            #if hasattr(a, "anchor") and a.anchor is not None:
            #    anchor = a.anchor

            #from ipdb import set_trace as dbg
            #dbg()

            #pos -= np.array(a.GetPosition()) + a.anchor - np.array(a.GetCenter())
            #pos += -np.array(a.GetCenter()) + a.GetPosition() - anchor
            border = np.array([pos + (offset[0], offset[1], 0),
                               pos + (-offset[0], offset[1], 0),
                               pos + (-offset[0], -offset[1], 0),
                               pos + (offset[0], -offset[1], 0),
                               pos + (offset[0], offset[1], 0)])

            borders.append(border)

        return actor.line(borders, colors=color, linewidth=linewidth)


class RelativeLayout(Layout):
    def __init__(self):
        self.constraints = {}
        self.actor_stack = []

    def add_constraint(self, actor, actor_anchor, parent, parent_anchor, dist):
        self.constraints[actor] = (actor_anchor, parent, parent_anchor, dist)

    # def _get_anchor_position(self, actor, anchor):
    #     x1, x2, y1, y2, z1, z2 = actor.GetBounds()
    #     width, height, depth = x2-x1, y2-y1, z2-z1

    #     if anchor == "center":
    #         return np.array(actor.GetCenter())
    #     elif anchor == "north-west":

    #         return np.array(())

    def _process_actor(self, actor, anchor=np.zeros(3)):
        if actor in self.actor_stack:
            raise RuntimeError("You have circular constraint(s).")

        self.actor_stack.append(actor)

        if actor in self.constraints:
            actor_anchor, parent, parent_anchor, dist = self.constraints[actor]
            parent_position = self._process_actor(parent, parent_anchor)

            position = parent_position + dist - (np.array(actor.GetCenter()) + actor_anchor)
        else:
            position = np.array(actor.GetCenter()) + anchor

        self.actor_stack.pop()

        return position

    def compute_positions(self, actors):
        positions = []
        for actor in actors:
            position = self._process_actor(actor)
            positions.append(position)

        return positions


class GridLayout2(Layout):
    def __init__(self, padding=0, cell_shape="rect", aspect_ratio=16/9., dim=None):
        self.cell_shape = cell_shape
        self.aspect_ratio = aspect_ratio
        self.dim = dim
        self.padding = (padding, padding) if type(padding) is int else padding

    def get_cells_shape(self, actors):
        if self.cell_shape == "rect":
            bounding_box_sizes = np.asarray(list(map(get_bounding_box_sizes, actors)))
            cell_shape = np.max(bounding_box_sizes, axis=0)[:2]
            shapes = [cell_shape] * len(actors)
        elif self.cell_shape == "square":
        #     bounding_box_sizes = np.asarray(list(map(get_bounding_box_sizes, actors)))
        #     cell_shape = np.max(bounding_box_sizes, axis=0)[:2]
        #     shapes = [(max(cell_shape),)*2] * len(actors)
        # elif self.cell_shape == "diagonal":
            # Size of every cell corresponds to the diagonal of the largest bounding box.
            longest_diagonal = np.max([a.GetLength() for a in actors])
            shapes = [(longest_diagonal, longest_diagonal)] * len(actors)
        else:
            raise ValueError("Unknown cell shape: '{0}'".format(self.cell_shape))

        return shapes

    def compute_positions(self, actors):
        shapes = self.get_cells_shape(actors)

        # Add margin around every cell.
        shapes = [np.array(self.padding)/2. + s for s in shapes]
        positions = get_grid_cells_position(shapes, self.aspect_ratio, self.dim)
        return positions

    def get_borders(self, actors, color=(1, 0, 0), linewidth=1):
        from dipy.viz import actor

        shapes = self.get_cells_shape(actors)
        positions = self.compute_positions(actors)

        print positions
        borders = []
        for pos, shape in zip(positions, shapes):
            offset = np.array(shape)/2.
            border = np.array([pos + (offset[0], offset[1], 0),
                               pos + (-offset[0], offset[1], 0),
                               pos + (-offset[0], -offset[1], 0),
                               pos + (offset[0], -offset[1], 0),
                               pos + (offset[0], offset[1], 0)])

            borders.append(border)

        return actor.line(borders, colors=color, linewidth=linewidth)
