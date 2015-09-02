from dipy.fixes import argparse as arg
import inspect

try:
    import numpydoc as ndoc
    has_ndoc = True
except ImportError:
    print('Numpydoc is not installed. \n pip install numpydoc')
    has_ndoc = False


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
        super(iap, self).__init__(prog, usage, description, epilog, version,
                                  parents, formatter_class, prefix_chars,
                                  fromfile_prefix_chars, argument_default,
                                  conflict_handler, add_help)

    def add_workflow(self, workflow):

        specs = inspect.getargspec(workflow)
        doc = inspect.getdoc(workflow)

        if has_ndoc:

            self.doc = ndoc.docscrape.NumpyDocString(doc)

        args = specs.args
        defaults = specs.defaults

        len_args = len(args)
        len_defaults = len(defaults)

        # Arguments with no defaults (Positional)
        cnt = 0
        for i in range(len_args - len_defaults):

            self.add_argument(args[i])
            cnt += 1

        # Arguments with defaults
        for i in range(cnt, len_args):
            self.add_argument('--' + args[i], help="ok")

    def update_argument(self, *args, **kargs):

        self.add_argument(*args, **kargs)