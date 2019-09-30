#!/bin/bash
set -ev

PIPI="pip install $EXTRA_PIP_FLAGS"

if [ -n "$USE_PRE" ]; then
    PIPI="$PIPI --find-links=$PRE_WHEELS --pre";
fi

$PIPI list
#---------- DIPY Installation -----------------

if [ "$INSTALL_TYPE" == "setup" ]; then
    python setup.py install
elif [ "$INSTALL_TYPE" == "pip" ]; then
    $PIPI .
elif [ "$INSTALL_TYPE" == "sdist" ]; then
    python setup_egg.py egg_info  # check egg_info while we're here
    python setup_egg.py sdist
    $PIPI dist/*.tar.gz
elif [ "$INSTALL_TYPE" == "wheel" ]; then
    pip install wheel
    python setup_egg.py bdist_wheel
    $PIPI dist/*.whl
elif [ "$INSTALL_TYPE" == "requirements" ]; then
    $PIPI -r requirements.txt
    python setup.py install
fi

# -------------- Run the tests -----------------

# Change into an innocuous directory and find tests from installation
mkdir for_testing
cd for_testing
# We need the setup.cfg for the pytest settings
cp ../setup.cfg .
# No figure windows for mpl; quote to hide : from travis-ci yaml parsing
echo "backend : agg" > matplotlibrc
if [ "${COVERAGE}" == "1" ]; then
    cp ../.coveragerc .;
    cp ../.codecov.yml .;
    COVER_CMD="coverage run -m ";
fi
$COVER_CMD pytest -s --doctest-modules --verbose --durations=10 --pyargs dipy