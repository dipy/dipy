"""Additional Command-line interface for spin."""
import os
import shutil

import click
from spin import util
from spin.cmds import meson


# From scipy: benchmarks/benchmarks/common.py
def _set_mem_rlimit(max_mem=None):
    """Set address space rlimit."""
    import resource

    import psutil

    mem = psutil.virtual_memory()

    if max_mem is None:
        max_mem = int(mem.total * 0.7)
    cur_limit = resource.getrlimit(resource.RLIMIT_AS)
    if cur_limit[0] > 0:
        max_mem = min(max_mem, cur_limit[0])

    try:
        resource.setrlimit(resource.RLIMIT_AS, (max_mem, cur_limit[1]))
    except ValueError:
        # on macOS may raise: current limit exceeds maximum limit
        pass


def _commit_to_sha(commit):
    p = util.run(['git', 'rev-parse', commit], output=False, echo=False)
    if p.returncode != 0:
        raise click.ClickException(
                f'Could not find SHA matching commit `{commit}`')

    return p.stdout.decode('ascii').strip()


def _dirty_git_working_dir():
    # Changes to the working directory
    p0 = util.run(['git', 'diff-files', '--quiet'])

    # Staged changes
    p1 = util.run(['git', 'diff-index', '--quiet', '--cached', 'HEAD'])

    return (p0.returncode != 0 or p1.returncode != 0)


def _run_asv(cmd):
    # Always use ccache, if installed
    PATH = os.environ['PATH']
    # EXTRA_PATH = os.pathsep.join([
    #     '/usr/lib/ccache', '/usr/lib/f90cache',
    #     '/usr/local/lib/ccache', '/usr/local/lib/f90cache'
    # ])
    env = os.environ
    env['PATH'] = f'EXTRA_PATH:{PATH}'

    # Control BLAS/LAPACK threads
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'

    # Limit memory usage
    try:
        _set_mem_rlimit()
    except (ImportError, RuntimeError):
        pass

    util.run(cmd, cwd='benchmarks', env=env)


@click.command()
@click.option(
    '--tests', '-t',
    default=None, metavar='TESTS', multiple=True,
    help="Which tests to run"
)
@click.option(
    '--compare', '-c',
    is_flag=True,
    default=False,
    help="Compare benchmarks between the current branch and main "
         "(unless other branches specified). "
         "The benchmarks are each executed in a new isolated "
         "environment."
)
@click.option(
    '--verbose', '-v', is_flag=True, default=False
)
@click.option(
    '--quick', '-q', is_flag=True, default=False,
    help="Run each benchmark only once (timings won't be accurate)"
)
@click.argument(
    'commits', metavar='',
    required=False,
    nargs=-1
)
@click.pass_context
def bench(ctx, tests, compare, verbose, quick, commits):
    """🏋 Run benchmarks.

    \b
    Examples:

    \b
    $ spin bench -t bench_lib
    $ spin bench -t bench_random.Random
    $ spin bench -t Random -t Shuffle

    Two benchmark runs can be compared.
    By default, `HEAD` is compared to `main`.
    You can also specify the branches/commits to compare:

    \b
    $ spin bench --compare
    $ spin bench --compare main
    $ spin bench --compare main HEAD

    You can also choose which benchmarks to run in comparison mode:

    $ spin bench -t Random --compare
    """
    if not commits:
        commits = ('main', 'HEAD')
    elif len(commits) == 1:
        commits = commits + ('HEAD',)
    elif len(commits) > 2:
        raise click.ClickException(
            'Need a maximum of two revisions to compare'
        )

    bench_args = []
    for t in tests:
        bench_args += ['--bench', t]

    if verbose:
        bench_args = ['-v'] + bench_args

    if quick:
        bench_args = ['--quick'] + bench_args

    if not compare:
        # No comparison requested; we build and benchmark the current version

        click.secho(
            "Invoking `build` prior to running benchmarks:",
            bold=True, fg="bright_green"
        )
        ctx.invoke(meson.build)

        meson._set_pythonpath()

        p = util.run(
            ['python', '-c', 'import dipy; print(dipy.__version__)'],
            cwd='benchmarks',
            echo=False,
            output=False
        )
        os.chdir('..')

        dipy_ver = p.stdout.strip().decode('ascii')
        click.secho(
            f'Running benchmarks on DIPY {dipy_ver}',
            bold=True, fg="bright_green"
        )
        cmd = [
            'asv', 'run', '--dry-run', '--show-stderr', '--python=same'
        ] + bench_args
        _run_asv(cmd)
    else:
        # Ensure that we don't have uncommited changes
        commit_a, commit_b = [_commit_to_sha(c) for c in commits]

        if commit_b == 'HEAD' and _dirty_git_working_dir():
            click.secho(
                "WARNING: you have uncommitted changes --- "
                "these will NOT be benchmarked!",
                fg="red"
            )

        cmd_compare = [
            'asv', 'continuous', '--factor', '1.05',
        ] + bench_args + [commit_a, commit_b]
        _run_asv(cmd_compare)


@click.command()
def clean():
    """🧹 Remove build and install folder."""
    build_dir = "build"
    install_dir = "build-install"
    print(f"Removing `{build_dir}`")
    if os.path.isdir(build_dir):
            shutil.rmtree(build_dir)
    print(f"Removing `{install_dir}`")
    if os.path.isdir(install_dir):
        shutil.rmtree(install_dir)