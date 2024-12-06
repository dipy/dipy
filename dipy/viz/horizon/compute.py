from threading import Thread


def compute(target, inputs):
    threads = []
    results = []

    def execute_target(data):
        result = target(data)
        results.append(result)

    for data in inputs:
        thread = Thread(target=execute_target, args=(data,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return results


def compute_volume(target, prev_idx, next_idx, intensities):
    results = []

    def execute(prev_idx, next_idx, intensities):
        result = target(prev_idx, next_idx, intensities)
        results.append(result)

    thread = Thread(target=execute, args=(prev_idx, next_idx, intensities,))
    thread.start()
    thread.join()

    return results[0]
