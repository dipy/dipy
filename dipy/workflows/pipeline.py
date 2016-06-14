from workflow import Workflow


class Pipeline(Workflow):
    def __init__(self, output_strategy='append', mix_names=False,
                 force=False, skip=False):

        self._optionals = {}
        super(Pipeline, self).__init__(output_strategy, mix_names, force, skip)

    def get_sub_runs(self):
        sub_runs = []
        for flow in self._get_sub_flows():
            sub_runs.append((flow.__name__, flow.run))

        return sub_runs

    def _get_sub_flows(self):
        raise AttributeError('Error: _get_sub_flows() needs to be defined for {}'.
                             format(self.__class__))

    def set_sub_flows_optionals(self, opts):
        self._optionals = {}
        for key, sub_dict in opts.iteritems():
            self._optionals[key] = \
                dict((k, v) for k, v in sub_dict.iteritems() if v is not None)

    def get_optionals(self, flow, **kwargs):
        opts = self._optionals[flow.__name__]
        opts.update(kwargs)

        return opts

    def run_sub_flow(self, flow, *args, **kwargs):
        flow.run(*args, **self.get_optionals(type(flow), **kwargs))
