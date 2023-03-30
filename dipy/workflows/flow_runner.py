
#  Disabling the FutureWarning from h5py below.
#  This disables the FutureWarning warning for all the workflows.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import logging

import click

from dipy import __version__ as dipy_version
from dipy.workflows.base import IntrospectiveArgumentParser, CustomCommand


def get_log_levels():
    """ Transforms the logging level passed on the commandline into a proper
    logging level name.
    """
    if sys.version_info[0] >= 3:
        return logging._nameToLevel
    else:
        return logging._levelNames


def run_flow(flow):
    """ Wraps the process of building an argparser that reflects the workflow
    that we want to run along with some generic parameters like logging,
    force and output strategies. The resulting parameters are then fed to
    the workflow's run method.
    """

    @click.command(cls=CustomCommand)
    @click.option("--force", is_flag=True, help='Force overwriting output files.')
    @click.option("--out-strat", help='Strategy to manage output creation.', default='absolute')
    @click.option("--mix-names", is_flag=True, help='Prepend mixed input names to output names.')
    @click.option(
        "--log-level", help='Log messages display level.',
        type=click.Choice(get_log_levels().keys(), case_sensitive=False))
    @click.option("--log-file", help='Log file to be saved.', type=click.Path())
    @click.version_option(f'DIPY {dipy_version}')
    def main(**kwargs):
        for k, v in kwargs.items():
            print(f"{k}: {v}")

        # TODO
        # args = parser.get_flow_args()
        #
        # logging.basicConfig(filename=args['log_file'],
        #                     format='%(levelname)s:%(message)s',
        #                     level=get_level(args['log_level']))
        #
        # # Output management parameters
        # flow._force_overwrite = args['force']
        # flow._output_strategy = args['out_strat']
        # flow._mix_names = args['mix_names']
        #
        # # Keep only workflow related parameters
        # del args['force']
        # del args['log_level']
        # del args['log_file']
        # del args['out_strat']
        # del args['mix_names']
        #
        # # Remove subflows related params
        # for params_dict in list(sub_flows_dicts.values()):
        #     for key in list(params_dict.keys()):
        #         if key in args.keys():
        #             params_dict[key] = args.pop(key)
        #
        #             # Rename dictionary key to the original param name
        #             params_dict[key.split('.')[1]] = params_dict.pop(key)
        #
        # if sub_flows_dicts:
        #     flow.set_sub_flows_optionals(sub_flows_dicts)
        #
        # flow.run(**args)

    parser = IntrospectiveArgumentParser(main)

    # TODO
    sub_flows_dicts = parser.add_workflow(flow)

    main()
