from warnings import warn
import numpy as np
import vtk
from vtk import vtkPoints, vtkCellArray, vtkPolyLine, vtkPolyData, vtkPolyDataWriter, vtkDoubleArray

class vtkstreamlinewriter:
    def __init__(self, streamlines=None, affine=np.eye(4)):
        """ Constructor

        Constructor initializes all variables

        Parameters
        ----------
        affine : numpy.array
            3D affine transformation

        Returns
        -------
        nothing

        Notes
        -----
        is called by self.clear()

        """
        self._affine = affine.copy() if affine is not None else np.eye(4)
        if self._check_streamlines(streamlines):
            self._streamlines = streamlines
        else:
            self._streamlines = None
        self._vertex_values = dict()
        self._streamline_values = dict()


    def get_number_of_streamlines(self):
        if self._streamlines is None:
            return 0
        else:
            return len(self._streamlines)


    def get_vertex_value_names(self):
        return self._vertex_values.keys()


    def get_streamline_value_names(self):
        return self._streamline_values.keys()



    def set_affine(self, affine=np.eye(4)):
        """ Sets the affine transformation for the streamlines

        The streamlines are expected and stored in voxel coordinates.
        The affine transformation matrix allows to save the streamlines in world coordinates.

        Parameters
        ----------
        affine : np.array
            4x4 array describing an affine transformation matrix

        Returns
        -------
        boolean
            True if the given matrix is of shape 4x4
            Falase if the matrix has an invalid shape

        Notes
        -----
        sets the identity matrix if the given parameter is 'None'

        """
        if affine is not None:
            if not affine.shape == (4, 4):
                return False
            self._affine = affine.copy()
            return True
        else:
            self._affine = np.eye(4)
            return True


    def get_affine(self):
        return self._affine.copy()


    def save(self, file_name, ascii=False, save_verts=False, save_vertex_values=True, save_streamline_values=True, save_colors=True, ignore_affine=False):
        """ Saves the streamlines

        Saves the streamlines in legacy VTK PolyData format.

        Parameters
        ----------
        file_name : string
            name of the vtk file
        ascii : boolean
            if 'True': saves in ascii mode
            if 'False' saves binary file
        save_verts : boolean
            if 'True': saves vertices with the streamlines
            if 'False' does not save vertices
        save_vertex_values : boolean
            if 'True': saves scalar values for vertices
            if 'False' does not vertex values
        save_streamline_values : boolean
            if 'True': saves scalar values for streamlines
            if 'False' does not streamline values
        save_colors : boolean
            if 'True': saves color codes of local and global direction of the streamlines
            if 'False' saves streamlines without directional color coding
        ignore_affine : boolean
            if 'True': saves streamlines in voxel coordinates
            if 'False' saves streamlines transformed with the affine matrix


        Returns
        -------
        boolean
            'True' if file was written successfully
            'False' if file could not be saved

        Notes
        -----
        I do not know how to catch when vtkPolyDataWriter fails to actually write the file...

        """

        if file_name == None:
            return False
        if len(file_name) == 0:
            return False
        if not isinstance(file_name, basestring):
            return False

        if self._streamlines is None:
            return False


        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        verts = vtk.vtkCellArray()

        for streamline in self._streamlines:
            line = vtkPolyLine()
            verts.InsertNextCell(len(streamline))
            for coord in streamline:
                if ignore_affine:
                    id = points.InsertNextPoint(coord)
                else:
                    cc = np.append(np.array(coord, float), 0)
                    id = points.InsertNextPoint(np.dot(self._affine, cc)[0:3])
                line.GetPointIds().InsertNextId(id)
                verts.InsertCellPoint(id)
            lines.InsertNextCell(line)

        # create polydata object and add lines (and vertices)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        if save_verts:
            polydata.SetVerts(verts)
        polydata.Modified()
        if vtk.VTK_MAJOR_VERSION <= 5:
            polydata.Update()


        # add color-coding of directions
        if save_colors:
            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)
            colors.SetName("Global_direction")
            for sl in self._streamlines:
                direction = np.absolute(np.asarray(sl[-1]) - np.asarray(sl[0]))
                direction /= np.linalg.norm(direction)
                direction *= 255
                colors.InsertNextTuple3(np.asscalar(direction[0]), np.asscalar(direction[1]), np.asscalar(direction[2]))
            polydata.GetCellData().SetScalars(colors)
            polydata.Modified()

            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)
            colors.SetName("Local_direction")
            for sl in self._streamlines:
                for i in xrange(0, len(sl)):
                    if i == 0:
                        direction = np.absolute(np.asarray(sl[1]) - np.asarray(sl[0]))
                    else:
                        direction = np.absolute(np.asarray(sl[i]) - np.asarray(sl[i-1]))
                    direction /= np.linalg.norm(direction)
                    direction *= 255
                    colors.InsertNextTuple3(np.asscalar(direction[0]), np.asscalar(direction[1]), np.asscalar(direction[2]))
            polydata.GetPointData().SetScalars(colors)
            polydata.Modified()



        # Add vertex values to polydata object
        if save_vertex_values and len(self._vertex_values.keys()) > 0:
            for name in self._vertex_values:
                values = vtk.vtkDoubleArray()
                values.SetNumberOfComponents(1)
                values.SetName(name)
                for sl in self._vertex_values[ name ]:
                    for i in sl:
                        values.InsertNextTuple1(np.asscalar(i))
                polydata.GetPointData().AddArray(values)
                polydata.Modified()



        # Add streamline values to polydata object
        if save_streamline_values and len(self._streamline_values.keys()) > 0:
            for name in self._streamline_values:
                values = vtk.vtkDoubleArray()
                values.SetNumberOfComponents(1)
                values.SetName(name)
                for i in self._streamline_values[ name ]:
                    values.InsertNextTuple1(np.asscalar(i))
                polydata.GetCellData().AddArray(values)
                polydata.Modified()




        # save polydata object
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(file_name)
        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(polydata)
        else:
            writer.SetInputData(polydata)

        if ascii:
            writer.SetFileTypeToASCII()
        else:
            writer.SetFileTypeToBinary()

        writer.Write()

        return True


    def decorate(self, data, name):
        """ This function adds values to the streamlines

        This function adds values to the streamlines.
        Values can be given for every vertex or for a complete streamline.

        To add vertex values the data array is a 3D image and vertices obtain
        the value of the voxels at their position. There is no interpolation,
        the value of the closest voxel is taken.
        This is useful to decorate the streamlines with e.g. FA values.

        To add streamline values the data array has 1 dimension and exactly the
        same number of entries as streamlines. The first streamline will obtain
        the first value in the array, the second streamline the second value, etc.
        This is useful to decorate the streamlines with e.g. the streamline's length.

        All values are stored as float64.

        The name identifies the values. It must be unique for vertex or streamline
        values within the category but it is possible (although not advisable) to
        have the same name for vertex and streamline values.

        Parameters
        ----------
        data : numpy.array
            3 dimensions for vertex values (3D image data)
            1 dimension for streamline values

        Returns
        -------
        boolean
            True if decorations were added successfully
            False if decorations were not added successfully

        Notes
        -----
        Possibly provide a selection of interpolation methods for vertex values

        """

        # we do need streamlines that we can decorate
        if self._streamlines is None:
            return False

        # ensure that input data is an array
        if type(data) != np.ndarray:
            return False

        # obtain vertex data from 3D image
        if len(data.shape) == 3:
            dim_x = data.shape[0]
            dim_y = data.shape[1]
            dim_z = data.shape[2]

            vert_vals = list()
            for i in range(0, len(self._streamlines)):
                streamline = self._streamlines[i]
                line_vals = np.zeros(len(streamline), np.float64)
                for j in range(0, len(streamline)):
                    coord = streamline[j]
                    if (0 <= coord[0] < dim_x) and (0 <= coord[1] < dim_y) and (0 <= coord[2] < dim_z):
                        line_vals[j] = data[ coord[0], coord[1], coord[2] ]
                    else:
                        line_vals[j] = np.float64('NaN')
                vert_vals.append(line_vals)

            self._vertex_values[ name ] = vert_vals
            return True

        # obtain streamline data from directly from input data
        if len(data.shape) == 1:
            # ensure that the length of input data corresponds with the number of streamlines
            if len(data) == self.get_number_of_streamlines():
                self._streamline_values[ name ] = data.copy()
            else:
                return False
            return True

        # input data has wrong shape
        return False




    def set_streamlines(self, streamlines):
        """ sets the streamlines to write

        Sets the given streamlines. Replaces eventually existing old streamlines
        (does not add but replace) and deletes any decorations

        Parameters
        ----------
        streamlines : list(numpy.ndarry(n, 3)
            List of n streamlines that are stored as numpy.ndarrays of shape (n, 3)

        Returns
        -------
        name : boolean
            True if the streamlines were successfully set
            False otherwise

        Notes
        -----
        Checks streamlines for correct type and format

        """


        if not self._check_streamlines(streamlines):
            return False

        self._streamlines = streamlines

        if len(self._vertex_values.keys()) != 0:
            self._vertex_values = dict()

        if len(self._streamline_values.keys()) != 0:
            self._streamline_values = dict()

        return True



    def _check_streamlines(self, streamlines):
        """ checks if streamlines have the right type

        checks if the streamlines are a list of numpy.ndarry of shape (n, 3)

        Parameters
        ----------
        streamlines : list(numpy.ndarry(n, 3)
            List of streamlines that are stored as numpy.ndarrays of shape (n, 3)

        Returns
        -------
        name : boolean
            True if the streamlines are as expected
            False otherwise

        Notes
        -----
        This is a rather thorough check that loops through all the streamlines.

        """
        if streamlines is None:
            return False

        if type(streamlines) is not list:
            return False
        for sl in streamlines:
            if sl.shape[1] is not 3:
                return False
        return True


