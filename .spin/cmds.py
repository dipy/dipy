"""Additional Command-line interface for spin."""
import os
import sys

import click
from spin import util
from spin.cmds import meson


@click.command()
@click.argument("asv_args", nargs=-1)
def asv(asv_args):
    """🏃 Run `asv` to collect benchmarks.

    ASV_ARGS are passed through directly to asv, e.g.:

    spin asv -- dev -b TransformSuite

    Please see CONTRIBUTING.txt
    """
    site_path = meson._get_site_packages()
    if site_path is None:
        print("No built scikit-image found; run `spin build` first.")
        sys.exit(1)

    python_path = f'{site_path}{os.sep}:{os.environ.get("PYTHONPATH", "")}'
    os.environ['PYTHONPATH'] = python_path
    util.run(['asv'] + list(asv_args))
