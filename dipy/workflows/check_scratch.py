###################
#This is the modified workflow script motivated from the example given on the following link.
#http://nipy.org/dipy/examples_built/workflow_creation.html#example-workflow-creation
#This script now accepts a folder path and lists all the files inside that folder.

#Current Testing Status
#The script has been tested with a sample file-path on my local DIPY installation and it
#behaves as expected. There is proper intimation of whether the file exist or not.

#Future Work:
#Currently the both the folder and the file path can be specified by the user. 

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
        the files present in the direcotory of the DIPY installation. 

        Parameters
        ----------
        input_files : string
            The path to the folder on the system (where DIPY has been installed). 

        out_dir : string
            Path to the output dir. Though this is not required for this script but since it is
            necessary parameter so keeping it here.
        """
        io_it = self.get_io_iterator()
        for fpath in io_it:
            if os.path.isdir(fpath):
            	file_list = os.listdir(fpath)
            	file_list = "\n".join(file_list)
                logging.info("The contents are as follows: \n"+file_list)
            else:
                logging.info(" The specified path does not exist.")



if __name__ == "__main__":
    run_flow(CheckScratch())



