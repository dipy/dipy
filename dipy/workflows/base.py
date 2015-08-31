from dipy.fixes import argparse as arg
import inspect


class BaseArgumentParser(arg.ArgumentParser):

    def set_workflow(self, workflow):

        specs = inspect.getargspec(workflow)
        args = specs.args
        defaults = specs.defaults

        len_args = len(args)
        len_defaults = len(defaults)

        print(len_args)
        print(len_defaults)

        # Arguments with no defaults
        cnt = 0
        for i in range(len_args - len_defaults):

            self.add_argument(args[i])
            cnt += 1

        # Arguments with defaults
        for i in range(cnt, len_args):
            self.add_argument('--' + args[i])




