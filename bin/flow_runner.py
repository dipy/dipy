from __future__ import division, print_function, absolute_import
from dipy.workflows.base import IntrospectiveArgumentParser

def run_flow(flow):
    parser = IntrospectiveArgumentParser()
    parser.add_workflow(flow)
    args = parser.get_flow_args()
    flow(**args)
