from dipy.workflows.workflow import Workflow
from dipy.workflows.combined_workflow import CombinedWorkflow


class DummyWorkflow1(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'dwf1'

    def run(self, inputs, param1=1, out_dir='', output_1='out1.txt'):
        """ Workflow used to test combined workflows in general.

        Parameters
        ----------
        inputs : string
            fake input string param
        param1 : int
            fake positional param (default 1)
        out_dir : string
            fake output directory (default '')
        out_combined : string
            fake out file (default out_combined.txt)

        References
        -----------
        dummy references
        """
        return param1


class DummyWorkflow2(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'dwf2'

    def run(self, inputs, param2=2, out_dir='', output_1='out2.txt'):
        """ Workflow used to test combined workflows in general.

        Parameters
        ----------
        inputs : string
            fake input string param
        param2 : int
            fake positional param (default 2)
        out_dir : string
            fake output directory (default '')
        out_combined : string
            fake out file (default out_combined.txt)
        """
        return param2


class DummyCombinedWorkflow(CombinedWorkflow):
    def _get_sub_flows(self):
        return [DummyWorkflow1, DummyWorkflow2]

    def run(self, inputs, param_combined=3, out_dir='',
            out_combined='out_combined.txt'):
        """ Workflow used to test combined workflows in general.

        Parameters
        ----------
        inputs : string
            fake input string param
        param_combined : int
            fake positional param (default 3)
        out_dir : string
            fake output directory (default '')
        out_combined : string
            fake out file (default out_combined.txt)
        """
        dwf1 = DummyWorkflow1()
        param1 = self.run_sub_flow(dwf1, inputs)

        dwf2 = DummyWorkflow2()
        param2 = self.run_sub_flow(dwf2, inputs)

        return param1, param2, param_combined


class TestFlow(Workflow):

    def run(self, positional_str, positional_bool, positional_int,
            positional_float, optional_str='default', optional_bool=False,
            optional_int=0, optional_float=1.0, optional_float_2=2.0,
            out_dir=''):
        """ Workflow used to test the introspective argument parser.

        Parameters
        ----------
        positional_str : string
            positional string argument
        positional_bool : bool
            positional bool argument
        positional_int : int
            positional int argument
        positional_float : float
            positional float argument
        optional_str : string, optional
            optional string argument (default 'default')
        optional_bool : bool, optional
            optional bool argument (default False)
        optional_int : int, optional
            optional int argument (default 0)
        optional_float : float, optional
            optional float argument (default 1.0)
        optional_float_2 : float, optional
            optional float argument #2 (default 2.0)
        out_dir : string
            output directory (default '')
        """
        return positional_str, positional_bool, positional_int,\
               positional_float, optional_str, optional_bool,\
               optional_int, optional_float, optional_float_2
