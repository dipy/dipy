=====================
🚀 DIPY Benchmarks 📊
=====================

Benchmarking DIPY with `Airspeed Velocity (ASV) <https://asv.readthedocs.io/>`__.
Measure the speed and performance of DIPY functions easily!

Prerequisites ⚙️
-----------------

Make sure you have the required tools installed:

.. code-block:: bash

    pip install spin asv virtualenv

Getting Started 🏃
-------------------

DIPY benchmarking uses ``spin``, which handles building DIPY and running ASV
automatically. You do not need to manually install a development version of
DIPY into your current Python environment.

Running Benchmarks 📈
---------------------

To run all available benchmarks, navigate to the root DIPY directory and run:

.. code-block:: bash

    spin bench

This builds DIPY and runs all benchmarks in the ``benchmarks/benchmarks/``
directory. Each benchmark is run multiple times to measure execution time
distribution — be patient, this can take a while.

For quick local testing (each benchmark runs only once, timings less accurate):

.. code-block:: bash

    spin bench --quick

To run benchmarks from a specific module, such as ``bench_segment.py``:

.. code-block:: bash

    spin bench -t bench_segment

To run a specific benchmark class, such as ``BenchQuickbundles``:

.. code-block:: bash

    spin bench -t bench_segment.BenchQuickbundles

To run benchmarks matching a pattern directly with ASV:

.. code-block:: bash

    cd benchmarks/
    asv run --dry-run --show-stderr --python=same --quick -b "bench.*Segment"

Comparing Results 📊
--------------------

To compare benchmark results between the current branch and ``master``:

.. code-block:: bash

    spin bench --compare
    spin bench --compare master
    spin bench --compare master HEAD

To compare a specific benchmark only:

.. code-block:: bash

    spin bench -t bench_segment --compare

To save results for future comparisons and view them in a browser:

.. code-block:: bash

    cd benchmarks/
    asv run -n -e --python=same
    asv publish
    asv preview

Continuous Integration 🤖
--------------------------

Benchmarks run automatically on every push and pull request via the
``Benchmarks / Linux`` CI check (see ``.github/workflows/benchmark.yml``).

The CI workflow:

- Installs DIPY with all dependencies
- Sets single-threaded environment variables for reliable timings
- Runs ``asv run`` against the current commit

.. note::

    Benchmark results are not yet published to a public dashboard.
    Contributions to set up ASV gh-pages publishing are welcome!

Contributing 🤝
---------------

Want to add or improve a benchmark? Here's how:

1. **Fork and clone** the DIPY repository.

2. **Create a new branch**:

   .. code-block:: bash

       git checkout -b bench/my-new-benchmark

3. **Add your benchmark** in ``benchmarks/benchmarks/``. Follow the naming
   convention ``bench_<module>.py`` (e.g., ``bench_tracking.py``).

4. **Test your benchmark locally**:

   .. code-block:: bash

       spin bench -t bench_mymodule --quick

5. **Open a pull request** with a description of what you are benchmarking
   and why it is useful to track performance.

Writing Benchmarks ✏️
---------------------

See the `ASV documentation <https://asv.readthedocs.io/>`__ for full details.

Key guidelines:

- The benchmark suite must be importable across multiple DIPY versions.
- Benchmark parameters must not depend on which DIPY version is installed.
- Keep individual benchmark runtimes reasonable (a few seconds at most).
- Use ASV's ``time_`` prefix for timing benchmarks, ``mem_`` for memory usage.
- Prepare large arrays and fixtures in ``setup()`` rather than in ``time_``
  methods, so setup cost is not included in the timing.
- Avoid benchmarks that require network access or large file downloads.

Example benchmark:

.. code-block:: python

    import numpy as np

    class BenchMyFunction:
        def setup(self):
            from dipy.data import get_fnames
            from dipy.io.streamline import load_tractogram
            fname = get_fnames(name="fornix")
            self.streamlines = load_tractogram(
                fname, "same", bbox_valid_check=False
            ).streamlines

        def time_my_function(self):
            from dipy.segment.clustering import QuickBundles
            qb = QuickBundles(threshold=10.0)
            qb.cluster(self.streamlines)

Embrace the Speed! ⏩
---------------------

You are all set to benchmark DIPY with ASV. Happy benchmarking! 🚀
