from dipy.workflows.workflow import Workflow


class CombinedWorkflow(Workflow):
    def __init__(self, output_strategy='append', mix_names=False,
                 force=False, skip=False):
        """ Workflow that combines multiple workflows.
        The workflow combined together are referred as sub flows in this class.
        """

        self._optionals = {}
        super(CombinedWorkflow, self).__init__(output_strategy, mix_names,
                                               force, skip)

    def get_sub_runs(self):
        """ Returns a list of tuples
        (sub flow name, sub flow run method, sub flow short name)
        to be used in the sub flow parameters extraction.
        """
        sub_runs = []
        for flow in self._get_sub_flows():
            sub_runs.append((flow.__name__, flow.run, flow.get_short_name()))

        return sub_runs

    def _get_sub_flows(self):
        """ Returns a list of sub flows used in the combined_workflow. Needs to
        be implemented in every new combined_workflow.
        """
        raise AttributeError('Error: _get_sub_flows() has to be defined for {}'.
                             format(self.__class__))

    def set_sub_flows_optionals(self, opts):
        """ Sets the self._optionals variable with all sub flow arguments
        that were passed in the commandline.
        """
        self._optionals = {}
        for key, sub_dict in opts.items():
            self._optionals[key] = \
                dict((k, v) for k, v in sub_dict.items() if v is not None)

    def get_optionals(self, flow, **kwargs):
        """ Returns the sub flow's optional arguments merged with those passed
        as params in kwargs.
        """
        opts = self._optionals[flow.__name__]
        opts.update(kwargs)

        return opts

    def run_sub_flow(self, flow, *args, **kwargs):
        """ Runs the sub flow with the optional parameters passed via the
        command line. This is a convenience method to make sub flow running
        more intuitive on the concrete CombinedWorkflow side.
        """
        return flow.run(*args, **self.get_optionals(type(flow), **kwargs))
