#!python

from dipy.workflows.flow_runner import run_flow
from dipy.workflows.vis_registration import VisualizeRegisteredImage

if __name__ == "__main__":
    run_flow(VisualizeRegisteredImage())