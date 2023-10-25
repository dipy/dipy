#  Disabling the FutureWarning from h5py below.
#  This disables the FutureWarning warning for all the workflows.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging

from dipy import __version__ as dipy_version
from dipy.workflows.base import IntrospectiveArgumentParser


def get_level(lvl):
    """ Transforms the logging level passed on the commandline into a proper
    logging level name.
    """
    try:
        return logging._levelNames[lvl]
    except Exception:
        return logging.INFO


def run_flow(flow):
    """ Wraps the process of building an argparser that reflects the workflow
    that we want to run along with some generic parameters like logging,
    force and output strategies. The resulting parameters are then fed to
    the workflow's run method.
    """
    parser = IntrospectiveArgumentParser()
    sub_flows_dicts = parser.add_workflow(flow)

    # Common workflow arguments
    parser.add_argument('--force', dest='force',
                        action='store_true', default=False,
                        help='Force overwriting output files.')

    parser.add_argument('--version', action='version',
                        version='DIPY {}'.format(dipy_version))

    parser.add_argument('--out_strat', action='store', dest='out_strat',
                        metavar='string', required=False, default='absolute',
                        help='Strategy to manage output creation.')

    parser.add_argument('--mix_names', dest='mix_names',
                        action='store_true', default=False,
                        help='Prepend mixed input names to output names.')

    # Add logging parameters common to all workflows
    msg = 'Log messages display level. Accepted options include CRITICAL,'
    msg += ' ERROR, WARNING, INFO, DEBUG and NOTSET (default INFO).'
    parser.add_argument('--log_level', action='store', dest='log_level',
                        metavar='string', required=False, default='INFO',
                        help=msg)

    parser.add_argument('--log_file', action='store', dest='log_file',
                        metavar='string', required=False, default='',
                        help='Log file to be saved.')

    args = parser.get_flow_args()

    logging.basicConfig(filename=args['log_file'],
                        format='%(levelname)s:%(message)s',
                        level=get_level(args['log_level']))

    # Output management parameters
    flow._force_overwrite = args['force']
    flow._output_strategy = args['out_strat']
    flow._mix_names = args['mix_names']

    # Keep only workflow related parameters
    del args['force']
    del args['log_level']
    del args['log_file']
    del args['out_strat']
    del args['mix_names']

    # Remove subflows related params
    for params_dict in list(sub_flows_dicts.values()):
        for key in list(params_dict.keys()):
            if key in args.keys():
                params_dict[key] = args.pop(key)

                # Rename dictionary key to the original param name
                params_dict[key.split('.')[1]] = params_dict.pop(key)

    if sub_flows_dicts:
        flow.set_sub_flows_optionals(sub_flows_dicts)

    return flow.run(**args)
