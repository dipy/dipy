"""Run a workflow stage in a fresh subprocess.

Workflows built on PyTorch load the ``libomp`` bundled with the torch wheel.
When DIPY's OpenMP extensions have already loaded another copy of ``libomp``
in the same process, torch can crash while loading model weights. Running
such stages in a spawned subprocess keeps the two runtimes apart.

This module must stay free of heavy imports: the spawned child imports it to
find :func:`stage_worker`, and anything imported here is loaded in the child
before the workflow itself.
"""

import importlib
import multiprocessing as mp
import os


def stage_worker(*, queue, module_name, class_name, init_params, run_params):
    """Run a workflow inside a spawned subprocess.

    Parameters
    ----------
    queue : multiprocessing.Queue
        Used to pass ``last_generated_outputs`` back to the parent.
    module_name : str
        Dotted module path of the workflow.
    class_name : str
        Class name of the workflow inside ``module_name``.
    init_params : dict
        Keyword arguments forwarded to ``workflow.__init__``.
    run_params : dict
        Keyword arguments forwarded to ``workflow.run``.
    """
    # Both copies are LLVM libomp, so letting a second one initialise is safe;
    # it must be allowed before the workflow import loads either of them.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    module = importlib.import_module(module_name)
    workflow = getattr(module, class_name)(**init_params)
    workflow.run(**run_params)
    queue.put(getattr(workflow, "last_generated_outputs", None))


def execute_stage_isolated(*, module_name, class_name, init_params, run_params):
    """Run a workflow stage in a spawned subprocess.

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
        If the child process exits with a non-zero exit code.
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
        },
    )
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(
            f"Isolated stage subprocess exited with code {proc.exitcode}"
        )
    return queue.get_nowait()
