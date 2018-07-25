#!python

"""
Importing the flow runner method and the VisualizeRegisteredImage.
"""

from dipy.workflows.flow_runner import run_flow
from dipy.workflows.vis_registeration import VisualizeRegisteredImage

"""
This is the method that will wrap everything that is needed to make a flow
command line ready then run it.
"""

if __name__ == "__main__":
    run_flow(VisualizeRegisteredImage())
