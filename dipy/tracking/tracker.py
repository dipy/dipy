from dipy.tracking.fast_tracking import generate_tractogram


def probabilistic_directions():
    pass


def deterministic_directions():
    pass


def eudx_directions():
    pass

def build_stopping_criterion(mask=None, check_point=None, include=None, exclude=None):
    sc_element = {
        "mask": mask,
        "wm_map": None,
        "gm_map": None,
        "csf_map": None,
        "checkpoint": check_point,
        "include": include,
        "exclude": exclude
    }
    return sc_element



def generate_streamlines():
    pass


def generate_tractogram_py(seeds, streamline_generator=generate_streamlines, postprocess_streamline=None):
    for s in seeds:
        streamline  = streamline_generator(s, stopping_criterion, direction_getter, max_cross, step_size, maxlen, fixedstep)
        if postprocess_streamline is not None:
            streamline  = postprocess_streamline(streamlines)
        yield streamline


def build_tractogram_algorithm(streamline_generator, tracker_func, pmf_gen_func,
                               stopping_criterion_func, postprocess_streamline=None):

    def generate_tractogram(seed_positions, seed_directions, tracking_parameters):
        for s in len(seed_positions):
            streamline  = streamline_generator(seed_position[i], seed_directions[i],
                                               tracker_func, pmf_gen_func,
                                               stopping_criterion_func)
            if postprocess_streamline is not None:
                streamline  = postprocess_streamline(streamlines)
            yield streamline

    return generate_tractogram


def local_tracking(seed_positons, seed_directions, *, tracking_parameters=None, tractogram_func=None,  postprocess_streamline=None):
    tractogram_func = tractogram_func or generate_tractogram
    if tractogram_func and 'cython' in type(tractogram_func):
        return tractogram_func(seed_positons, seed_directions, tracking_parameters)

    return generate_tractogram_py(seed_positons, seed_directions, tracking_parameters)


probabilistic_tracking = partial(local_tracking, tractogram_func=generate_tractogram)
probabilistic_tracking_python = partial(local_tracking, tractogram_func=generate_tractogram_py)

deterministic_tracking = partial(local_tracking, direction_getter='deterministic')
eudx_tracking = partial(local_tracking, direction_getter='eudx')
fact_tracking = partial(local_tracking, direction_getter='fact')
ptt_tracking = partial(local_tracking, direction_getter='ptt')
pft_tracking = partial(local_tracking, direction_getter='pft')


def test_probabilistic_tracking():
    pass


if __name__ == "__main__":
    my_generate_tractogram = build_tractogram_algorithm()
    streamlines = local_tracking(p_seed, d_seed, params, tractogram_func=my_generate_tractogram)
