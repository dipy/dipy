from dipy.fixes import argparse as arg

import inspect

from dipy.workflows.docstring_parser import NumpyDocString

class IntrospectiveArgumentParser(arg.ArgumentParser):

    def __init__(self, prog=None, usage=None, description=None, epilog=None,
                 version=None, parents=[],
                 formatter_class=arg.RawTextHelpFormatter,
                 prefix_chars='-', fromfile_prefix_chars=None,
                 argument_default=None, conflict_handler='resolve',
                 add_help=True):

        """ Augmenting the argument parser to allow automatic creation of
        arguments from workflows

        Parameters
        -----------
        prog : None
            The name of the program (default: sys.argv[0])
        usage : None
            A usage message (default: auto-generated from arguments)
        description : str
            A description of what the program does
        epilog : str
            Text following the argument descriptions
        version : None
            Add a -v/--version option with the given version string
        parents : list
            Parsers whose arguments should be copied into this one
        formatter_class : obj
            HelpFormatter class for printing help messages
        prefix_chars : str
            Characters that prefix optional arguments
        fromfile_prefix_chars : None
            Characters that prefix files containing additional arguments
        argument_default : None
            The default value for all arguments
        conflict_handler : str
            String indicating how to handle conflicts
        add_help : bool
            Add a -h/-help option
        """

        iap = IntrospectiveArgumentParser
        if epilog is None:
            epilog = \
                "Garyfallidis, E., M. Brett, B. Amirbekian, A. Rokem, S. Van Der " \
                "Walt, M. Descoteaux, and I. Nimmo-Smith. Dipy, a library for the " \
                "analysis of diffusion MRI data. Frontiers in Neuroinformatics," \
                " 1-18, 2014."

        super(iap, self).__init__(prog, usage, description, epilog, version,
                                  parents, formatter_class, prefix_chars,
                                  fromfile_prefix_chars, argument_default,
                                  conflict_handler, add_help)

        self.doc = None

    def add_workflow(self, workflow):
        specs = inspect.getargspec(workflow)
        doc = inspect.getdoc(workflow)
        self.doc = NumpyDocString(doc)['Parameters']

        self.outputs = NumpyDocString(doc)['Outputs']

        args = specs.args
        defaults = specs.defaults

        len_args = len(args)
        len_defaults = len(defaults)

        for i, arg in enumerate(args):
            prefix = ''
            is_optionnal = i >= len_args - len_defaults
            if is_optionnal:
                prefix = '--'

            typestr = self.doc[i][1]
            dtype, isnarg = self._select_dtype(typestr)
            help_msg = ''.join(self.doc[i][2])

            _args = ['{0}{1}'.format(prefix, arg)]
            _kwargs = {'help': help_msg,
                       'type': dtype,
                       'action': 'store'}

            if is_optionnal:
               _kwargs["metavar"] = dtype.__name__

            if dtype is bool:
                _kwargs['type'] = int
                _kwargs['choices'] = [0, 1]

            if isnarg:
                 _kwargs['nargs'] = '*'

            self.add_argument(*_args, **_kwargs)

    def _select_dtype(self, text):
        text = text.lower()
        nargs_str = 'variable'
        is_nargs = nargs_str in text
        arg_type = None

        if 'str' in text:
            arg_type = str
        if 'int' in text:
            arg_type = int
        if 'float' in text:
            arg_type = float
        if 'bool' in text:
            arg_type = bool

        return arg_type, is_nargs

    def get_flow_args(self, args=None, namespace=None):
        ns_args = self.parse_args(args, namespace)
        dct = vars(ns_args)

        return dict((k, v) for k, v in dct.items() if v is not None)

    def update_argument(self, *args, **kargs):
        self.add_argument(*args, **kargs)

    def show_argument(self, dest):
        for act in self._actions[1:]:
            if act.dest == dest:
                print(act)

    def add_epilogue(self):
        pass

    def add_description(self):
        pass

    def get_outputs(self):
        return self.outputs

