from __future__ import division, print_function, absolute_import
import logging

from dipy.workflows.base import IntrospectiveArgumentParser

def run_flow(flow):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    parser = IntrospectiveArgumentParser()
    parser.add_workflow(flow)
    args = parser.get_flow_args()
    flow(**args)
