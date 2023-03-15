import itertools
import sys
import inspect

import argparse
import collections
from dipy.workflows.docstring_parser import NumpyDocString


Param = collections.namedtuple(
    "Param", ("name", "type", "description", "default"))

if sys.version_info[0] >= 3:
    Empty = inspect.Parameter.empty
else:
    class Empty:
        """Marker object for args with no default values
        (as opposed to args defaulting to None).

        Copied from the `inspect._empty` class in Python 3.
        """


def get_args_default(func):
    """Return a dict mapping function parameter names to their default values
    or :class:`Empty` if no default value is specified."""
    arg_defaults = {}
    if sys.version_info[0] >= 3:
        sig_object = inspect.signature(func)
        for param in sig_object.parameters.values():
            if param.name != "self":
                arg_defaults[param.name] = param.default
    else:
        specs = inspect.getargspec(func)
        names = reversed(specs.args[1:])
        defaults = reversed(specs.defaults)
        params = itertools.zip_longest(names, defaults, fillvalue=Empty)
        for name, default in params:
            arg_defaults[name] = default

    return arg_defaults


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

        self._output_params = None
        self._positional_params = None
        self._optional_params = None

        # Flag to keep track of whether a variable
        # required arg is already present.
        self._variable_arg_present = None

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
        # get the args from the doc string
        # doc_parameters will be a list of tuples of (arg name, type, description)
        npds = NumpyDocString(inspect.getdoc(workflow.run))
        self.description = "{0}\n\n{1}".format(
            " ".join(npds["Summary"]),
            " ".join(npds["Extended Summary"]))

        # get the arg default values from the actual function definition
        arg_defaults = get_args_default(workflow.run)

        self._compile_args(npds["Parameters"], arg_defaults)

        # insert any references into "References" section of the epilog
        if npds["References"]:
            ref_header = "References: \n"
            ref_text = ''.join((text or "\n" for text in npds['References']))
            if ref_header not in self.epilog:
                self.epilog += ref_header + ref_text
            else:
                epilog_parts = self.epilog.split(ref_header, 1)
                self.epilog = ''.join((
                    epilog_parts[0], ref_header, ref_text, "\n", epilog_parts[0]
                ))

        # add required parameters
        self._variable_arg_present = False
        for param in self._positional_params:
            self._add_positional_arg(param, self)

        # add optional parameters
        for param in self._optional_params:
            self._add_optional_arg(param, self)

        # add output parameters
        if self._output_params:
            output_args = self.add_argument_group('output arguments (optional)')
            for param in self._output_params:
                self._add_optional_arg(param, output_args)

        # TODO refactor add_sub_flow_args
        # return self.add_sub_flow_args(workflow.get_sub_runs())
        return {}

    def _compile_args(self, docstring_parameters, arg_defaults):
        """Sort the parameters into positional args, optional args, and outputs
        based on the parameter names and default values."""
        self._positional_params = []
        self._optional_params = []
        self._output_params = []
        for param_name, param_type, param_desc in docstring_parameters:
            if param_name not in arg_defaults:
                raise ValueError(
                    f"Argument '{param_name}' not present "
                    f"in function signature.")

            description = " ".join(param_desc)
            default_value = arg_defaults.pop(param_name)
            # TODO remove for backwards compatibility?
            param = Param(param_name, param_type, description, default_value)

            if param_name.startswith("out_"):
                if default_value is Empty:
                    raise ValueError(
                        f"Required argument '{param_name}' "
                        f"starts with reserved keyword 'out'.")
                self._output_params.append(param)
            elif "optional" in param_type:
                if default_value is Empty:
                    raise ValueError(
                        f"Required argument '{param_name}' "
                        f"marked as optional in docstring.")
                self._optional_params.append(param)
            elif default_value is Empty:
                self._positional_params.append(param)
            else:
                raise ValueError(
                    f"Arg '{param_name}' has default value but is not "
                    f"not marked as output or optional in docstring.")

        # Ensure that number of function parameters match
        # the arguments in the doc strings
        if arg_defaults:
            raise ValueError(
                f"Arguments are missing from docstring: {list(arg_defaults)}")

    def _add_positional_arg(self, param, arg_group):
        """Add a new positional argument to the parser."""
        dtype, isnarg = self._select_dtype(param.type)
        _kwargs = {"help": param.description, "action": "store"}

        if dtype is bool:
            _kwargs["type"] = none_or_dtype(int)
            _kwargs["choices"] = [0, 1]
        elif dtype is tuple:
            _kwargs["type"] = none_or_dtype(str)
        else:
            _kwargs["type"] = none_or_dtype(dtype)

        if isnarg:
            if self._variable_arg_present:
                raise ValueError(
                    f"{self.prog}: All positional arguments present "
                    "are gathered into a list. It does not make "
                    "much sense to have more than one positional "
                    "argument with 'variable string' as dtype. "
                    "Please, ensure that 'variable (type)' "
                    "appears only once as a positional argument. ")
            _kwargs["nargs"] = "+"
            self._variable_arg_present = True

        # arg_name = f"{param.name}".replace("_", "-")
        arg_name = param.name
        arg_group.add_argument(arg_name, **_kwargs)

    def _add_optional_arg(self, param, arg_group):
        """Add a new optional argument to the parser."""
        dtype, isnarg = self._select_dtype(param.type)
        _kwargs = {"help": param.description, "action": "store"}

        if dtype is bool:
            _kwargs["action"] = "store_true" if param.default else "store_false"
        else:
            if dtype is tuple:
                dtype = str
            _kwargs["type"] = none_or_dtype(dtype)
            _kwargs["metavar"] = dtype.__name__

        if isnarg:
            _kwargs["nargs"] = '*'

        # arg_name = f"--{param.name}".replace("_", "-")
        arg_name = f"--{param.name}"
        arg_group.add_argument(arg_name, **_kwargs)

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

    @staticmethod
    def _select_dtype(arg_type_str):
        """ Analyses a docstring parameter line and returns the corresponding
        data type for the argparse argument.

        Parameters
        ----------
        arg_type_str : string
            Parameter text line to inspect.

        Returns
        -------
        arg_type : The type found by inspecting the text line.

        is_nargs : Whether this argument is nargs
        (arparse's multiple values argument)
        """
        arg_type_str = arg_type_str.lower()
        is_nargs = "variable" in arg_type_str

        if "str" in arg_type_str:
            arg_type = str
        elif "int" in arg_type_str:
            arg_type = int
        elif "float" in arg_type_str:
            arg_type = float
        elif "bool" in arg_type_str:
            arg_type = bool
        elif "tuple" in arg_type_str:
            arg_type = tuple
        else:
            raise ValueError(
                f"Unable to determine data type from docstring: {arg_type_str}")

        return arg_type, is_nargs

    def get_flow_args(self, args=None, namespace=None):
        """Return the parsed arguments as a dictionary that will be used
        as a workflow's run method arguments.
        """
        ns_args = self.parse_args(args, namespace)
        flow_args = {
            arg_name: value if value != "None" else None
            for arg_name, value in vars(ns_args).items()
            if value is not None
        }
        return flow_args

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
