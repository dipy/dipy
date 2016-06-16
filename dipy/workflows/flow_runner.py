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
    sub_flows_dicts = parser.add_workflow(flow)

    # Common workflow arguments
    parser.add_argument('--force', dest='force',
                        action='store_true', default=False,
                        help='Force overwriting output files.')

    parser.add_argument('--out_strat', action='store', dest='out_strat',
                        metavar='string', required=False, default='append',
                        help='Strategy to manage output creation.')

    parser.add_argument('--mix_names', dest='mix_names',
                        action='store_true', default=False,
                        help='Prepend mixed input names to output names.')

    # Add logging parameters common to all workflows
    parser.add_argument('--log_level', action='store', dest='log_level',
                        metavar='string', required=False, default='INFO',
                        help='Log messsages display level')

    parser.add_argument('--log_file', action='store', dest='log_file',
                        metavar='string', required=False, default='',
                        help='Log file to be saved.')

    args = parser.get_flow_args()

    logging.basicConfig(filename=args['log_file'],
                        format='%(levelname)s:%(message)s',
                        level=get_level(args['log_level']))

    flow._output_strategy = args['out_strat']
    flow._mix_names = args['mix_names']

    # Keep only workflow related parameters
    del args['force']
    del args['log_level']
    del args['log_file']
    del args['out_strat']
    del args['mix_names']

    # Remove subflows related params
    for sub_flow, params_dict in sub_flows_dicts.iteritems():
        for key, _ in params_dict.iteritems():
            if key in args.keys():
                params_dict[key] = args.pop(key)

    if sub_flows_dicts:
        flow.set_sub_flows_optionals(sub_flows_dicts)

    flow.run(**args)

