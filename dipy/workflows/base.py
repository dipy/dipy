from dipy.fixes import argparse as arg
import inspect


class AugmentedArgumentParser(arg.ArgumentParser):

    def add_workflow(self, workflow):

        specs = inspect.getargspec(workflow)
        #print(specs)

        args = specs.args
        defaults = specs.defaults


        len_args = len(args)
        len_defaults = len(defaults)

        #print(len_args)
        #print(len_defaults)

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



#    def update_argument(self, option_string1, option_string2, dest,
#                        nargs=None, const=None, default=None, type=None,
#                        choices=None, help=None, metavar=None):
#
#        for i, act in enumerate(self._actions):
#            act_tmp = act
#            #self._actions[i] = arg._StoreAction(option_strings, dest,
#            #                                    nargs, const, default, type,
#            #                                    choices, help, metavar)
#            self.add_argument(option_strings1, dest, nargs, const,
#                              default, type,
#                              choices, help, metavar)

