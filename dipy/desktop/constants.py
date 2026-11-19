"""Names, formats and launch command shared by the Skyline desktop modules."""

from dipy.utils.assets import asset_home

APP_NAME = "DIPY SKYLINE"
SKYLINE_URL_SCHEME = "dipy-skyline"
ASSET_APP = "dipy-skyline"
SKYLINE_HOME = asset_home(ASSET_APP)
APP_LOG_FILE = SKYLINE_HOME / "skyline_app.log"
MODULE_NAME = "dipy.desktop.launch"
SKYLINE_PACKAGES = ("fury", "imgui_bundle")
# Files per launch, report request, API call and IPC message, and per
# ``dipy-skyline://`` link; the confirmation dialog names every file of a link.
MAX_FILES = 64
MAX_LINK_FILES = 64
# Files named in a window title.
SUMMARY_FILES = 3
# Launched from a menu, a link or a report, nothing activated the environment:
# its folders are missing from PATH, and an editable install cannot find the
# ``ninja`` it needs to import DIPY. The code holds no double quote or percent
# sign, so it survives the registry, shortcut and desktop entry quoting.
LAUNCH_BOOTSTRAP = (
    "import os,sys,runpy;"
    "d=os.path.dirname(sys.executable);"
    "os.environ['PATH']=os.pathsep.join([d,os.path.join(d,'Scripts'),"
    "os.path.join(d,'Library','bin'),os.environ.get('PATH','')]);"
    f"runpy.run_module('{MODULE_NAME}',run_name='__main__')"
)
# Python puts the current folder first on ``sys.path`` when it runs ``-c``, and
# the folder of a file opened from the file manager may hold a ``runpy.py`` or
# a ``dipy`` package that someone else planted: ``-P`` keeps it off the path.
PYTHON_SAFE_PATH = "-P"
# Formats the Skyline viewer can load; anything else opens in the browser.
SKYLINE_FORMATS = (
    ".nii",
    ".nii.gz",
    ".trx",
    ".trk",
    ".tck",
    ".pam5",
    ".vtk",
    ".vtp",
    ".fib",
    ".dpy",
    ".nrrd",
    ".mgh",
    ".mgz",
    ".gii",
    ".gii.gz",
    ".pial",
)
