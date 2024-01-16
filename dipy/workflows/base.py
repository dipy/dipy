import argparse
import inspect

from dipy.workflows.docstring_parser import NumpyDocString


def get_args_default(func):
    sig_object = inspect.signature(func)
    params = sig_object.parameters.values()
    names = [param.name for param in params if param.name != 'self']
    defaults = [param.default for param in params
                if param.default is not inspect._empty]

    return names, defaults


def none_or_dtype(dtype):
    """Check None presence before type casting."""
    local_type = dtype

    def inner(value):
        if value in ['None', 'none']:
            return 'None'
        return local_type(value)
    return inner


class IntrospectiveArgumentParser(argparse.ArgumentParser):

    def __init__(self, prog=None, usage=None, description=None, epilog=None,
                 parents=(), formatter_class=argparse.RawTextHelpFormatter,
                 prefix_chars='-', fromfile_prefix_chars=None,
                 argument_default=None, conflict_handler='resolve',
                 add_help=True):
        """ Augmenting the argument parser to allow automatic creation of
        arguments from workflows

        Parameters
        ----------
        prog : None
            The name of the program. (default: sys.argv[0])
        usage : None
            A usage message. (default: auto-generated from arguments)
        description : str
            A description of what the program does.
        epilog : str
            Text following the argument descriptions.
        parents : list
            Parsers whose arguments should be copied into this one.
        formatter_class : obj
            HelpFormatter class for printing help messages.
        prefix_chars : str
            Characters that prefix optional arguments.
        fromfile_prefix_chars : None
            Characters that prefix files containing additional arguments.
        argument_default : None
            The default value for all arguments.
        conflict_handler : str
            String indicating how to handle conflicts.
        add_help : bool
            Add a -h/-help option.
        """

        iap = IntrospectiveArgumentParser
        if epilog is None:
            epilog =\
                ("References: \n"
                 "Garyfallidis, E., M. Brett, B. Amirbekian, A. Rokem,"
                 " S. Van Der Walt, M. Descoteaux, and I. Nimmo-Smith. Dipy, a"
                 " library for the analysis of diffusion MRI data. Frontiers"
                 " in Neuroinformatics, 1-18, 2014.")

        super(iap, self).__init__(prog=prog, usage=usage,
                                  description=description,
                                  epilog=epilog, parents=parents,
                                  formatter_class=formatter_class,
                                  prefix_chars=prefix_chars,
                                  fromfile_prefix_chars=fromfile_prefix_chars,
                                  argument_default=argument_default,
                                  conflict_handler=conflict_handler,
                                  add_help=add_help)

        self.doc = None

    def add_workflow(self, workflow):
        """Take a workflow object and use introspection to extract the
        parameters, types and docstrings of its run method. Then add these
        parameters to the current arparser's own params to parse. If the
        workflow is of type combined_workflow, the optional input parameters
        of its sub workflows will also be added.

        Parameters
        ----------
        workflow : dipy.workflows.workflow.Workflow
            Workflow from which to infer parameters.

        Returns
        -------
        sub_flow_optionals : dictionary of all sub workflow optional parameters
        """

        doc = inspect.getdoc(workflow.run)
        npds = NumpyDocString(doc)
        self.doc = npds['Parameters']
        self.description = '{0}\n\n{1}'.format(
            ' '.join(npds['Summary']),
            ' '.join(npds['Extended Summary']))

        if npds['References']:
            ref_text = [text or "\n" for text in npds['References']]
            ref_idx = self.epilog.find('References: \n') + \
                len('References: \n')
            self.epilog = "{0}{1}\n{2}".format(self.epilog[:ref_idx],
                                               ''.join(ref_text),
                                               self.epilog[ref_idx:])

        self._output_params = [param for param in npds['Parameters']
                               if 'out_' in param[0]]
        self._positional_params = [param for param in npds['Parameters']
                                   if 'optional' not in param[1] and
                                      'out_' not in param[0]]
        self._optional_params = [param for param in npds['Parameters']
                                 if 'optional' in param[1]]

        args, defaults = get_args_default(workflow.run)

        output_args = self.add_argument_group('output arguments(optional)')

        len_args = len(args)
        len_defaults = len(defaults)
        nb_positional_variable = 0

        if len_args != len(self.doc):
            raise ValueError(
                    self.prog + ": Number of parameters in the "
                    "doc string and run method does not match. "
                    "Please ensure that the number of parameters "
                    "in the run method is same as the doc string.")

        for i, arg in enumerate(args):
            prefix = ''
            is_optional = i >= len_args - len_defaults
            if is_optional:
                prefix = '--'

            typestr = self.doc[i][1]
            dtype, isnarg = self._select_dtype(typestr)
            help_msg = ' '.join(self.doc[i][2])

            _args = ['{0}{1}'.format(prefix, arg)]
            _kwargs = {'help': help_msg,
                       'type': dtype,
                       'action': 'store'}

            if is_optional:
                _kwargs['metavar'] = dtype.__name__
                if dtype is bool:
                    _kwargs['action'] = 'store_true'
                    default_ = dict()
                    default_[arg] = False
                    self.set_defaults(**default_)
                    del _kwargs['type']
                    del _kwargs['metavar']
            elif dtype is bool:
                _kwargs['type'] = int
                _kwargs['choices'] = [0, 1]

            if dtype is tuple:
                _kwargs['type'] = str

            if isnarg:
                if is_optional:
                    _kwargs['nargs'] = '*'
                else:
                    _kwargs['nargs'] = '+'
                    nb_positional_variable += 1

            if 'out_' in arg:
                output_args.add_argument(*_args, **_kwargs)
            else:
                if _kwargs['action'] != 'store_true':
                    _kwargs['type'] = none_or_dtype(_kwargs['type'])
                self.add_argument(*_args, **_kwargs)

        if nb_positional_variable > 1:
            raise ValueError(self.prog + " : All positional arguments present"
                             " are gathered into a list. It does not make"
                             "much sense to have more than one positional"
                             " argument with 'variable string' as dtype."
                             " Please, ensure that 'variable (type)'"
                             " appears only once as a positional argument."
                             )

        return self.add_sub_flow_args(workflow.get_sub_runs())

    def add_sub_flow_args(self, sub_flows):
        """ Take an array of workflow objects and use introspection to extract
        the parameters, types and docstrings of their run method. Only the
        optional input parameters are extracted for these as they are treated
        as sub workflows.

        Parameters
        ----------
        sub_flows : array of dipy.workflows.workflow.Workflow
            Workflows to inspect.

        Returns
        -------
        sub_flow_optionals : dictionary of all sub workflow optional parameters
        """

        sub_flow_optionals = dict()
        for name, flow, short_name in sub_flows:
            sub_flow_optionals[name] = {}
            doc = inspect.getdoc(flow)
            npds = NumpyDocString(doc)
            _doc = npds['Parameters']

            args, defaults = get_args_default(flow)

            len_args = len(args)
            len_defaults = len(defaults)

            flow_args = \
                self.add_argument_group('{0} arguments(optional)'.
                                        format(name))

            for i, arg_name in enumerate(args):
                is_not_optionnal = i < len_args - len_defaults
                if 'out_' in arg_name or is_not_optionnal:
                    continue

                arg_name = '{0}.{1}'.format(short_name, arg_name)
                sub_flow_optionals[name][arg_name] = None
                prefix = '--'
                typestr = _doc[i][1]
                dtype, isnarg = self._select_dtype(typestr)
                help_msg = ''.join(_doc[i][2])

                _args = ['{0}{1}'.format(prefix, arg_name)]
                _kwargs = {'help': help_msg, 'type': dtype, 'action': 'store',
                           'metavar': dtype.__name__}

                if dtype is bool:
                    _kwargs['action'] = 'store_true'
                    default_ = dict()
                    default_[arg_name] = False
                    self.set_defaults(**default_)
                    del _kwargs['type']
                    del _kwargs['metavar']
                elif dtype is bool:
                    _kwargs['type'] = int
                    _kwargs['choices'] = [0, 1]

                if dtype is tuple:
                    _kwargs['type'] = str

                if isnarg:
                    _kwargs['nargs'] = '*'

                if _kwargs['action'] != 'store_true':
                    _kwargs['type'] = none_or_dtype(_kwargs['type'])
                flow_args.add_argument(*_args, **_kwargs)

        return sub_flow_optionals

    def _select_dtype(self, text):
        """ Analyses a docstring parameter line and returns the good argparser
        type.

        Parameters
        ----------
        text : string
            Parameter text line to inspect.

        Returns
        -------
        arg_type : The type found by inspecting the text line.

        is_nargs : Whether or not this argument is nargs
        (arparse's multiple values argument)
        """
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
        if 'tuple' in text:
            arg_type = tuple

        return arg_type, is_nargs

    def get_flow_args(self, args=None, namespace=None):
        """Return the parsed arguments as a dictionary that will be used
        as a workflow's run method arguments.
        """

        ns_args = self.parse_args(args, namespace)
        dct = vars(ns_args)
        res = dict((k, v) for k, v in dct.items() if v is not None)
        res.update(dict((k, None) for k, v in res.items() if v == 'None'))
        return res

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

    @property
    def output_parameters(self):
        return self._output_params

    @property
    def positional_parameters(self):
        return self._positional_params

    @property
    def optional_parameters(self):
        return self._optional_params
