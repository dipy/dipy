#!/bin/bash
set -ev

PIPI="pip install $EXTRA_PIP_FLAGS"

if [ -n "$USE_PRE" ]; then
    PIPI="$PIPI --find-links=$PRE_WHEELS --pre";
fi

pip install --upgrade virtualenv
virtualenv $VENV_ARGS venv
source venv/bin/activate
# just to check python version
python --version

# Needed for Python 3.5 wheel fetching
$PIPI --upgrade pip setuptools

# We need nose for numpy.testing<=1.15
$PIPI pytest nose;
$PIPI numpy;
if [ -n "$DEPENDS" ]; then $PIPI $DEPENDS; fi
if [ "${COVERAGE}" == "1" ]; then pip install coverage coveralls codecov; fi
if [ "${VTK}" == "1" ]; then
    sudo apt-get update;
    sudo apt-get install -y $VTK_VER;
    sudo apt-get install -y xvfb;
    sudo apt-get install -y python-tk;
    sudo apt-get install -y python-imaging;
    $PIPI xvfbwrapper;
fi