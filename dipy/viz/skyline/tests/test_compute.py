import threading
import time

from dipy.viz.skyline.compute import process_async_callbacks, run_async


def _wait_for_results(results, expected, timeout=10.0):
    deadline = time.time() + timeout
    while len(results) < expected and time.time() < deadline:
        process_async_callbacks()
        time.sleep(0.005)
    process_async_callbacks()


def test_run_async_delivers_the_return_value_to_the_callback():
    results = []

    run_async(lambda a, b: a + b, lambda r, e: results.append((r, e)), 2, b=3)
    _wait_for_results(results, 1)

    assert results == [(5, None)]


def test_run_async_delivers_the_exception_to_the_callback():
    results = []

    def boom():
        raise RuntimeError("worker failed")

    run_async(boom, lambda r, e: results.append((r, e)))
    _wait_for_results(results, 1)

    assert len(results) == 1
    result, exception = results[0]
    assert result is None
    assert isinstance(exception, RuntimeError)
    assert str(exception) == "worker failed"


def test_run_async_runs_the_work_off_the_calling_thread():
    worker_threads = []

    run_async(threading.get_ident, lambda r, e: worker_threads.append(r))
    _wait_for_results(worker_threads, 1)

    assert worker_threads[0] != threading.get_ident()


def test_process_async_callbacks_runs_on_the_calling_thread():
    callback_threads = []
    done = threading.Event()

    def record(_result, _exception):
        callback_threads.append(threading.get_ident())
        done.set()

    run_async(lambda: None, record)
    while not done.is_set():
        process_async_callbacks()
        time.sleep(0.005)

    assert callback_threads == [threading.get_ident()]


def test_process_async_callbacks_is_a_noop_on_an_empty_queue():
    process_async_callbacks()

    calls = []
    run_async(lambda: "value", lambda r, e: calls.append(r))
    _wait_for_results(calls, 1)
    process_async_callbacks()

    assert calls == ["value"]


def test_process_async_callbacks_drains_every_queued_callback():
    results = []

    for index in range(5):
        run_async(lambda i=index: i * 10, lambda r, e: results.append(r))
    _wait_for_results(results, 5)

    assert sorted(results) == [0, 10, 20, 30, 40]


def test_run_async_uses_daemon_threads():
    before = {t.ident for t in threading.enumerate()}
    started = threading.Event()
    release = threading.Event()

    def slow():
        started.set()
        release.wait(5)

    run_async(slow, lambda r, e: None)
    started.wait(5)
    new_threads = [t for t in threading.enumerate() if t.ident not in before]

    assert new_threads
    assert all(t.daemon for t in new_threads)

    release.set()
    _wait_for_results([], 0)
