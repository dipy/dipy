###################
#This is a basic workflow script motivated from the example given on the following link.
#http://nipy.org/dipy/examples_built/workflow_creation.html#example-workflow-creation
#This script simply accepts a file path (as string) from the user and checks to see 
#if the specified files exists or not?

#Current Testing Status
#The script has been tested with a sample file-path on my local DIPY installation and it
#behaves as expected. There is proper intimation of whether the file exist or not.

#Future Work:
#Currently the file path can be specified by the user but in future more dynamic parameters such as
#specifying both the directory and the filepath will be implemented in the script.

#Doing the necessary imports.
import logging
import os.path

#Doing the imports for enabling DIPY workflows
from dipy.workflows.workflow import Workflow
from dipy.workflows.flow_runner import run_flow

#Declaring the class name here.
class CheckScratch(Workflow):

    def run(self, input_files,out_dir=''):
        """ Scratch direcotry probing method for validating the status of 
        the files present in the 'Very_scratch` direcotory of the DIPY installation.

        It checks the `scratch/very_scratch` directory within the DIPY installation
        for the specified file (from the user) and shows the appropriate message. 

        Parameters
        ----------
        input_files : string
            Path to the input file that need to be cheked within the
            the dipy installation.

        out_dir : string
            Path to the output dir. Though this is not required for this script but since it is
            necessary parameter so keeping it here.
        """
        io_it = self.get_io_iterator()
        for fpath in io_it:
            if os.path.isfile(fpath):
                logging.info(" The specified file exist in the very_scratch directory.")
            else:
                logging.info(" The file that you specified does not exist in the very_scratch directory")



if __name__ == "__main__":
    run_flow(CheckScratch())



