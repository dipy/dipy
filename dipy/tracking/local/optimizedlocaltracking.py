import numpy as np

from .localtrack import local_tracker
from dipy.align import Bunch
from dipy.tracking import utils

#import itertools
import multiprocessing
#import sys
#import os

"""
# Configure the environment
if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = '/Users/support/Documents/code/dependencies/spark' # NOTE: set you spark home path here

# Create a variable for our root path
SPARK_HOME = os.environ['SPARK_HOME']

# Add the PySpark/py4j to the Python Path
sys.path.insert(1, os.path.join(SPARK_HOME, "python", "build"))
sys.path.insert(1, os.path.join(SPARK_HOME, "python"))

from pyspark import  SparkContext
# Note: you can only run this code once before getting an error if you don't run sc.stop()
sc = SparkContext( 'local', 'pyspark')
"""

#nums = sc.parallelize(xrange(1000))
#def duh(x):
#    sdfjkalsdhaf
#new_nums = nums.map(duh)
#print str(new_nums.collect())

# enum TissueClass (tissue_classifier.pxd) is not accessible
# from here. To be changed when minimal cython version > 0.21.
# cython 0.21 - cpdef enum to export values into Python-level namespace
# https://github.com/cython/cython/commit/50133b5a91eea348eddaaad22a606a7fa1c7c457
TissueTypes = Bunch(OUTSIDEIMAGE=-1, INVALIDPOINT=0, TRACKPOINT=1, ENDPOINT=2)

#N = self.maxlen
#dg = self.direction_getter
#tc = self.tissue_classifier
#ss = self.step_size
#fixed = self.fixed
#max_cross = self.max_cross
#vs = self._voxel_size
#
#inv_A = np.linalg.inv(self.affine)
#lin = inv_A[:3, :3]
#offset = inv_A[:3, 3]
#
#F = np.empty((N + 1, 3), dtype=float)
#B = F.copy()
SEEDS = 0


# run normally
# see if specific arguments can be "pickled" or passed into Pool

def streamline_computation_wrapper(streamline_arguments):
    print "got in wrapper class: "
    print len(streamline_arguments.seed)
    print len(streamline_arguments.config_args)
    print
    return streamline_computation(streamline_arguments.seed, *streamline_arguments.config_args)

class StreamlineArguments():
    """
    Wrapper class to hold streamline arguments
    """
    def __init__(self, seed, config_args):
        self.seed = seed
        self.config_args = config_args


def streamline_computation(s, inv_A, lin, offset, F, B, vs, dg, tc, affine,
                 ss, max_cross=None, maxlen=500, fixed=True,
                 return_all=True):
    """
    Helper function for parallelizing the computation of streamlines
    """
    print "got in wrapper class: "

    global SEEDS
    SEEDS += 1
    print "at seed " + str(SEEDS)    
    
    s = np.dot(lin, s) + offset
    directions = dg.initial_direction(s)
    directions = directions[:max_cross]
    for first_step in directions:
        stepsF, tissue_class = local_tracker(dg, tc, s, first_step,
                                             vs, F, ss, fixed)
        if not (return_all or
                tissue_class == TissueTypes.ENDPOINT or
                tissue_class == TissueTypes.OUTSIDEIMAGE):
            continue
        first_step = -first_step
        stepsB, tissue_class = local_tracker(dg, tc, s, first_step,
                                             vs, B, ss, fixed)
        if not (return_all or
                tissue_class == TissueTypes.ENDPOINT or
                tissue_class == TissueTypes.OUTSIDEIMAGE):
            continue

        if stepsB == 1:
            streamline = F[:stepsF].copy()
        else:
            parts = (B[stepsB-1:0:-1], F[:stepsF])
            streamline = np.concatenate(parts, axis=0)
        return streamline

class OptimizedLocalTracking(object):
    """A streamline generator for local tracking methods"""
    """ 
    This class has the same functionality as LocakTracking except 
    it has an additional method which generates the streamlines in 
    bulk using pyspark
    """

    @staticmethod
    def _get_voxel_size(affine):
        """Computes the voxel sizes of an image from the affine.

        Checks that the affine does not have any shear because local_tracker
        assumes that the data is sampled on a regular grid.

        """
        lin = affine[:3, :3]
        dotlin = np.dot(lin.T, lin)
        # Check that the affine is well behaved
        if not np.allclose(np.triu(dotlin, 1), 0.):
            msg = ("The affine provided seems to contain shearing, data must "
                   "be acquired or interpolated on a regular grid to be used "
                   "with `LocalTracking`.")
            raise ValueError(msg)
        return np.sqrt(dotlin.diagonal())

    def __init__(self, direction_getter, tissue_classifier, seeds, affine,
                 step_size, max_cross=None, maxlen=500, fixedstep=True,
                 return_all=True):
        """Creates streamlines by using local fiber-tracking.

        Parameters
        ----------
        direction_getter : instance of DirectionGetter
            Used to get directions for fiber tracking.
        tissue_classifier : instance of TissueClassifier
            Identifies endpoints and invalid points to inform tracking.
        seeds : array (N, 3)
            Points to seed the tracking. Seed points should be given in point
            space of the track (see ``affine``).
        affine : array (4, 4)
            Coordinate space for the streamline point with respect to voxel
            indices of input data. This affine can contain scaling, rotational,
            and translational components but should not contain any shearing.
            An identity matrix can be used to generate streamlines in "voxel
            coordinates" as long as isotropic voxels were used to acquire the
            data.
        step_size : float
            Step size used for tracking.
        max_cross : int or None
            The maximum number of direction to track from each seed in crossing
            voxels. By default all initial directions are tracked.
        maxlen : int
            Maximum number of steps to track from seed. Used to prevent
            infinite loops.
        fixedstep : bool
            If true, a fixed stepsize is used, otherwise a variable step size
            is used.
        return_all : bool
            If true, return all generated streamlines, otherwise only
            streamlines reaching end points or exiting the image.
        """
        self.direction_getter = direction_getter
        self.tissue_classifier = tissue_classifier
        self.seeds = seeds
        if affine.shape != (4, 4):
            raise ValueError("affine should be a (4, 4) array.")
        self.affine = affine
        self._voxel_size = self._get_voxel_size(affine)
        self.step_size = step_size
        self.fixed = fixedstep
        self.max_cross = max_cross
        self.maxlen = maxlen
        self.return_all = return_all

    
    def pickle_test():
        pass
    
    def __iter__(self):
        # Make tracks, move them to point space and return
        track = self._generate_streamlines()
        return utils.move_streamlines(track, self.affine)

    def _generate_streamlines(self):
        """A streamline generator"""
        N = self.maxlen
        dg = self.direction_getter
        tc = self.tissue_classifier
        ss = self.step_size
        fixed = self.fixed
        max_cross = self.max_cross
        vs = self._voxel_size

        # Get inverse transform (lin/offset) for seeds
        inv_A = np.linalg.inv(self.affine)
        lin = inv_A[:3, :3]
        offset = inv_A[:3, 3]

        F = np.empty((N + 1, 3), dtype=float)
        B = F.copy()
        for s in self.seeds:
            s = np.dot(lin, s) + offset
            directions = dg.initial_direction(s)
            directions = directions[:max_cross]
            for first_step in directions:
                stepsF, tissue_class = local_tracker(dg, tc, s, first_step,
                                                     vs, F, ss, fixed)
                if not (self.return_all or
                        tissue_class == TissueTypes.ENDPOINT or
                        tissue_class == TissueTypes.OUTSIDEIMAGE):
                    continue
                first_step = -first_step
                stepsB, tissue_class = local_tracker(dg, tc, s, first_step,
                                                     vs, B, ss, fixed)
                if not (self.return_all or
                        tissue_class == TissueTypes.ENDPOINT or
                        tissue_class == TissueTypes.OUTSIDEIMAGE):
                    continue

                if stepsB == 1:
                    streamline = F[:stepsF].copy()
                else:
                    parts = (B[stepsB-1:0:-1], F[:stepsF])
                    streamline = np.concatenate(parts, axis=0)
                yield streamline
 
    @staticmethod
    def bunch_arguments(inv_A, lin, offset, F, B, vs, dg, tc, affine,
                 ss, max_cross=None, maxlen=500, fixed=True,
                 return_all=True):
        args = []
        args.append(inv_A)
        args.append(lin)
        args.append(offset)
        args.append(F)
        args.append(B)
        args.append(vs)
        args.append(dg)
        args.append(tc)
        args.append(affine)
        args.append(ss)
        args.append(max_cross)
        args.append(maxlen)
        args.append(fixed)
        args.append(return_all)
        return args

   
    def compute_all_streamlines(self):
        """computes all streamlines using Pool in Python multiprocessing library"""

        print "starting function"        
        
        N = self.maxlen
        dg = self.direction_getter
        tc = self.tissue_classifier
        ss = self.step_size
        fixed = self.fixed
        max_cross = self.max_cross
        vs = self._voxel_size

        # Get inverse transform (lin/offset) for seeds
        inv_A = np.linalg.inv(self.affine)
        lin = inv_A[:3, :3]
        offset = inv_A[:3, 3]

        F = np.empty((N + 1, 3), dtype=float)
        B = F.copy()
        
        print "about to bunch arguments"   
        
        # TODO: better bunch arguments by using *args
        arguments = self.bunch_arguments(inv_A, lin, offset, F, B, vs, dg, tc, self.affine,
                 ss, max_cross, N, fixed,
                 self.return_all)
        
        print "arguments bunched here they are: "
        print str(arguments)
        for i in range(0,3):
            print
        
        
        

        print "initialized pool about to do streamline"
        #args_lst = list(itertools.izip(self.seeds, itertools.repeat(arguments)))
        
#        for i in range(0,3):
#            print (str(args_lst[i]))
#            for i in range(0,3):
#                print
        
#        print "length of args_lst: "  
#        print str(len(args_lst))

        
        # TODO: use partials to do this in func_tools
        stream_args = list(StreamlineArguments(s, arguments) for s in self.seeds)                
#       stream_args = list(tc for s in self.seeds)

        #stream_args = list(1 for i in range(0,10))    
        #sc_stream_args = sc.parallelize(stream_args)            
        #streamlines = sc_stream_args.map(streamline_computation_wrapper)
       
        p = multiprocessing.Pool(multiprocessing.cpu_count())

        #results = p.map(check_if_tc_is_cool,stream_args)
        
        streamlines = p.map(streamline_computation_wrapper, stream_args) 
        
        
        #streamlines = []
        #for seed_with_parameters in args_lst:
#            print "seed"
#            print s
#            print
#            print
#            print
#            print str(args_lst)
#            for i in range(0,5):
#                print
        #    streamlines.append(streamline_computation_wrapper(StreamlineArguments(seed_with_parameters[0], seed_with_parameters[1])))
        
        #return streamlines
        return streamlines

def check_if_tc_is_cool(arg):
    print(arg)
    return 1

#def _get_voxel_size(affine):
#    """Computes the voxel sizes of an image from the affine.
#
#    Checks that the affine does not have any shear because local_tracker
#    assumes that the data is sampled on a regular grid.
#
#    """
#    lin = affine[:3, :3]
#    dotlin = np.dot(lin.T, lin)
#    # Check that the affine is well behaved
#    if not np.allclose(np.triu(dotlin, 1), 0.):
#        msg = ("The affine provided seems to contain shearing, data must "
#               "be acquired or interpolated on a regular grid to be used "
#               "with `LocalTracking`.")
#        raise ValueError(msg)
#    return np.sqrt(dotlin.diagonal())
#
#
#def compute_all_streamlines(direction_getter, tissue_classifier, seeds, affine,
#                 step_size, max_cross=None, maxlen=500, fixedstep=True,
#                 return_all=True):
#    """computes all streamlines using Pool in Python multiprocessing library"""
#    N = maxlen
#    dg = direction_getter
#    tc = tissue_classifier
#    ss = step_size
#    fixed = fixedstep
#    if affine.shape != (4, 4):
#        raise ValueError("affine should be a (4, 4) array.")
#    vs = _get_voxel_size(affine)
#    
#
#
#    # Get inverse transform (lin/offset) for seeds
#    inv_A = np.linalg.inv(affine)
#    lin = inv_A[:3, :3]
#    offset = inv_A[:3, 3]
#
#    F = np.empty((N + 1, 3), dtype=float)
#    B = F.copy()
#    
#    def streamline_computation(s):
#        """
#        Helper function for parallelizing the computation of streamlines
#        """
#        s = np.dot(lin, s) + offset
#        directions = dg.initial_direction(s)
#        directions = directions[:max_cross]
#        for first_step in directions:
#            stepsF, tissue_class = local_tracker(dg, tc, s, first_step,
#                                                 vs, F, ss, fixed)
#            if not (return_all or
#                    tissue_class == TissueTypes.ENDPOINT or
#                    tissue_class == TissueTypes.OUTSIDEIMAGE):
#                continue
#            first_step = -first_step
#            stepsB, tissue_class = local_tracker(dg, tc, s, first_step,
#                                                 vs, B, ss, fixed)
#            if not (return_all or
#                    tissue_class == TissueTypes.ENDPOINT or
#                    tissue_class == TissueTypes.OUTSIDEIMAGE):
#                continue
#    
#            if stepsB == 1:
#                streamline = F[:stepsF].copy()
#            else:
#                parts = (B[stepsB-1:0:-1], F[:stepsF])
#                streamline = np.concatenate(parts, axis=0)
#            return streamline
#            
#    p = multiprocessing.Pool(multiprocessing.cpu_count())
#    streamlines = p.map(streamline_computation, seeds) 
#    return streamlines
        
        
