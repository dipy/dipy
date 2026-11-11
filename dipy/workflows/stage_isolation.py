"""Run a workflow stage in a fresh subprocess.

Workflows built on PyTorch load the ``libomp`` bundled with the torch wheel.
When DIPY's OpenMP extensions have already loaded another copy of ``libomp``
in the same process, torch can crash while loading model weights. Running
such stages in a spawned subprocess keeps the two runtimes apart.

This module must stay free of heavy imports: the spawned child imports it to
find :func:`stage_worker`, and anything imported here is loaded in the child
before the workflow itself. :mod:`dipy.utils.logging` loads no compiled
extension.
"""

import importlib
import multiprocessing as mp
import os
from queue import Empty
import traceback

from dipy.utils.logging import add_file_handler, attached_log_files, logger

_POLL_SECONDS = 0.5


def stage_worker(
    *, queue, module_name, class_name, init_params, run_params, log_level, log_files
):
    """Run a workflow inside a spawned subprocess.

    The outcome is put on ``queue`` as a ``(status, payload)`` pair: ``("ok",
    last_generated_outputs)`` on success, ``("error", formatted_traceback)``
    when the workflow raises.

    Parameters
    ----------
    queue : multiprocessing.Queue
        Used to pass the outcome back to the parent.
    module_name : str
        Dotted module path of the workflow.
    class_name : str
        Class name of the workflow inside ``module_name``.
    init_params : dict
        Keyword arguments forwarded to ``workflow.__init__``.
    run_params : dict
        Keyword arguments forwarded to ``workflow.run``.
    log_level : int
        Level of the DIPY logger in the parent.
    log_files : list of tuple
        ``(filename, level)`` of the log files of the parent, which the
        child appends to.
    """
    # Both copies are LLVM libomp, so letting a second one initialise is safe;
    # it must be allowed before the workflow import loads either of them.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)
    for filename, level in log_files:
        add_file_handler(filename, level=level)

    try:
        module = importlib.import_module(module_name)
        workflow = getattr(module, class_name)(**init_params)
        workflow.run(**run_params)
        outcome = ("ok", getattr(workflow, "last_generated_outputs", None))
    except BaseException:
        error = traceback.format_exc()
        logger.error(f"Isolated stage {module_name}.{class_name} failed:\n{error}")
        outcome = ("error", error)

    queue.put(outcome)


def execute_stage_isolated(*, module_name, class_name, init_params, run_params):
    """Run a workflow stage in a spawned subprocess.

    The child logs to the same files and at the same level as the parent.

    Parameters
    ----------
    module_name : str
        Dotted module path of the workflow.
    class_name : str
        Class name of the workflow inside ``module_name``.
    init_params : dict
        Keyword arguments forwarded to ``workflow.__init__``.
    run_params : dict
        Keyword arguments forwarded to ``workflow.run``.

    Returns
    -------
    object
        The value of ``workflow.last_generated_outputs`` from the child
        process, or ``None`` if the attribute was absent.

    Raises
    ------
    RuntimeError
        If the workflow raises in the child, with the child traceback in the
        message, or if the child exits without reporting an outcome.
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=stage_worker,
        kwargs={
            "queue": queue,
            "module_name": module_name,
            "class_name": class_name,
            "init_params": init_params,
            "run_params": run_params,
            "log_level": logger.getEffectiveLevel(),
            "log_files": attached_log_files(),
        },
    )
    proc.start()

    # The result is read before joining: a child blocks on exit until the
    # queue is drained, so joining first deadlocks on large outputs.
    outcome = None
    while outcome is None:
        try:
            outcome = queue.get(timeout=_POLL_SECONDS)
        except Empty:
            if proc.is_alive():
                continue
            try:
                outcome = queue.get(timeout=_POLL_SECONDS)
            except Empty:
                proc.join()
                raise RuntimeError(
                    f"Isolated stage subprocess exited with code {proc.exitcode} "
                    "without reporting a result"
                ) from None

    proc.join()
    status, payload = outcome
    if status == "error":
        raise RuntimeError(f"Isolated stage subprocess failed:\n{payload}")
    return payload
