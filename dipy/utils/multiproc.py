"""Function for determining the effective number of processes to be used."""

from multiprocessing import cpu_count
import os
import subprocess
import sys
from warnings import warn


def determine_num_processes(num_processes):
    """Determine the effective number of processes for parallelization.

    - For `num_processes = None`` return the maximum number of cores retrieved
    by cpu_count().

    - For ``num_processes > 0``, return this value.

    - For ``num_processes < 0``, return the maximal number of cores minus
    ``num_processes + 1``. In particular ``num_processes = -1`` will use as
    many cores as possible.

    - For ``num_processes = 0`` a ValueError is raised.

    Parameters
    ----------
    num_processes : int or None
        Desired number of processes to be used.
    """
    if not isinstance(num_processes, int) and num_processes is not None:
        raise TypeError("num_processes must be an int or None")

    if num_processes == 0:
        raise ValueError("num_processes cannot be 0")

    try:
        if num_processes is None:
            return cpu_count()

        if num_processes < 0:
            return max(1, cpu_count() + num_processes + 1)
    except NotImplementedError:
        warn("Cannot determine number of cores. Using only 1.", stacklevel=2)
        return 1

    return num_processes


def get_usable_cpu_affinity():
    """Return the set of CPU core IDs available to the current process.

    ``os.cpu_count()`` reports every core on the machine, which overcounts on
    an HPC node or in a container where the process is pinned to a subset.
    Sizing a thread or process pool from it then oversubscribes the cores and
    slows the job down. This looks at the affinity mask where the platform
    exposes one, and falls back to the machine core count otherwise.

    Returns
    -------
    cpus : set of int
        Usable core IDs. Never empty.
    """
    # Native Unix/Linux affinity, honours cgroups and taskset.
    if hasattr(os, "sched_getaffinity"):
        try:
            return os.sched_getaffinity(0)
        except Exception:
            pass

    # Python 3.13+ multiplatform standard API.
    if hasattr(os, "process_cpu_count"):
        proc_count = os.process_cpu_count()
        if proc_count:
            return set(range(proc_count))

    # macOS: query the system core count.
    if sys.platform == "darwin":
        try:
            res = subprocess.check_output(["sysctl", "-n", "hw.ncpu"]).strip()
            return set(range(int(res)))
        except Exception:
            pass

    # Windows and generic fallback.
    try:
        return set(range(cpu_count()))
    except Exception:
        return set(range(os.cpu_count() or 1))
