=====================
🚀 DIPY Benchmarks 📊
=====================

Benchmarking Dipy with Airspeed Velocity (ASV). Measure the speed and performance of DIPY functions easily!

Prerequisites ⚙️
---------------------

Before you start, make sure you have ASV and installed:

.. code-block:: bash

    pip install asv
    pip install virtualenv

Getting Started 🏃‍♂️
---------------------

DIPY Benchmarking is as easy as a piece of 🍰 with ASV. You don't need to install a development version of DIPY into your current Python environment. ASV manages virtual environments and builds DIPY automatically.

Running Benchmarks 📈
---------------------

To run all available benchmarks, navigate to the root DIPY directory at the command line and execute:

.. code-block:: bash

    spin bench

This command builds DIPY and runs all available benchmarks defined in the ``benchmarks/`` directory. Be patient; this could take a while as each benchmark is run multiple times to measure execution time distribution.

For local testing without replications, unleash the power of ⚡:

.. code-block:: bash

    cd benchmarks/
    export REGEXP="bench.*Ufunc"
    asv run --dry-run --show-stderr --python=same --quick -b $REGEXP

Here, ``$REGEXP`` is a regular expression used to match benchmarks, and ``--quick`` is used to avoid repetitions.

To run benchmarks from a particular benchmark module, such as ``bench_segment.py``, simply append the filename without the extension:

.. code-block:: bash

    spin bench -t bench_segment

To run a benchmark defined in a class, such as ``BenchQuickbundles`` from ``bench_segment.py``, show your benchmarking ninja skills:

.. code-block:: bash

    spin bench -t bench_segment.BenchQuickbundles

Comparing Results 📊
--------------------

To compare benchmark results with another version/commit/branch, use the ``--compare`` option (or ``-c``):

.. code-block:: bash

    spin bench --compare v1.7.0 -t bench_segment
    spin bench --compare 20d03bcfd -t bench_segment
    spin bench -c master -t bench_segment

These commands display results in the console but don't save them for future comparisons. For greater control and to save results for future comparisons, use ASV commands:

.. code-block:: bash

    cd benchmarks
    asv run -n -e --python=same
    asv publish
    asv preview

Benchmarking Versions 💻
------------------------

To benchmark or visualize releases on different machines locally, generate tags with their commits:

.. code-block:: bash

    cd benchmarks
    # Get commits for tags
    # delete tag_commits.txt before re-runs
    for gtag in $(git tag --list --sort taggerdate | grep "^v"); do
    git log $gtag --oneline -n1 --decorate=no | awk '{print $1;}' >> tag_commits.txt
    done
    # Use the last 20 versions for maximum power 🔥
    tail --lines=20 tag_commits.txt > 20_vers.txt
    asv run HASHFILE:20_vers.txt
    # Publish and view
    asv publish
    asv preview

Contributing 🤝
---------------

TBD

Writing Benchmarks ✏️
---------------------

See `ASV documentation <https://asv.readthedocs.io/>`__ for basics on how to write benchmarks.

Things to consider:

- The benchmark suite should be importable with multiple DIPY version.
- Benchmark parameters should not depend on which DIPY version is installed.
- Keep the runtime of the benchmark reasonable.
- Prefer ASV's ``time_`` methods for benchmarking times.
- Prepare arrays in the setup method rather than in the ``time_`` methods.
- Be mindful of large arrays created.

Embrace the Speed! ⏩
---------------------

Now you're all set to benchmark DIPY with ASV and watch your code reach for the stars! Happy benchmarking! 🚀



