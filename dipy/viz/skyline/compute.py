import queue
import threading

_callback_queue = queue.Queue()


def run_async(func, callback, *args, **kwargs):
    """
    Execute a function asynchronously in a background thread.

    This function runs the provided function in a daemon thread and
    queues the callback to be executed on the main thread upon
    completion.

    Parameters
    ----------
    func : callable
        The function to execute in the background.
    callback : callable
        Function to call when execution completes. Must have signature:
        ``callback(result, exception)`` where result is the return value
        of func and exception is any exception raised during execution.
    *args : tuple
        Positional arguments to pass to func.
    **kwargs : dict
        Keyword arguments to pass to func.

    Notes
    -----
    The callback is not executed directly in the worker thread. Instead,
    it is queued for execution on the main thread via the callback queue.
    The worker thread is created as a daemon thread, so it will
    terminate if the main application closes.
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
    """Processes all finished tasks in the queue explicitly."""

    while not _callback_queue.empty():
        callback, result, exception = _callback_queue.get_nowait()

        callback(result, exception)
