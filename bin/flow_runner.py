from __future__ import division, print_function, absolute_import
import logging

from dipy.workflows.base import IntrospectiveArgumentParser

def get_level(lvl):
    try:
        return logging._levelNames[lvl]
    except:
        return logging.INFO

def run_flow(flow):
    parser = IntrospectiveArgumentParser()
    parser.add_workflow(flow)

    # Add logging parameters common to all workflows
    parser.add_argument('--log_level', action='store', dest='log_level',
                        metavar='string', required=False, default='INFO',
                        help='Log messsages display level')

    parser.add_argument('--log_file', action='store', dest='log_file',
                        metavar='string', required=False, default='',
                        help='Log messsages display level')

    args = parser.get_flow_args()
    logging.basicConfig(filename=args['log_file'],
                        format='%(levelname)s:%(message)s',
                        level=get_level(args['log_level']))
    # Keep only workflow related parameters
    del args['log_level']
    del args['log_file']

    flow(**args)
