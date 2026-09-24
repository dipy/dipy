"""Run callables in background threads with main-thread callback delivery."""

import queue
import threading

_callback_queue = queue.Queue()


def run_async(func, callback, *args, **kwargs):
    """Execute ``func`` asynchronously in a background daemon thread.

    Parameters
    ----------
    func : callable
        The function to execute in the background thread.
    callback : callable
        Function invoked as ``callback(result, exception)`` once ``func``
        completes, where ``result`` is its return value and ``exception``
        is any exception it raised, or None on success.
    *args
        Positional arguments passed to ``func``.
    **kwargs
        Keyword arguments passed to ``func``.

    Notes
    -----
    ``callback`` is never called from the worker thread. It is queued
    together with the result and exception, and only runs when
    :func:`process_async_callbacks` drains the queue on whichever thread
    calls it (normally the main/UI thread). The worker thread is a daemon
    thread, so it does not block interpreter shutdown.
    """

    def worker():
        result = None
        exception = None
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            exception = e
        finally:
            # We do NOT execute the callback here!
            # We package it up and send it to the main thread.
            _callback_queue.put((callback, result, exception))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


def process_async_callbacks():
    """Drain the async task queue and run each callback on the calling thread.

    Pair this with :func:`run_async` on the main/UI thread so background work
    delivers results safely without blocking the worker thread.
    """

    while not _callback_queue.empty():
        callback, result, exception = _callback_queue.get_nowait()

        callback(result, exception)
