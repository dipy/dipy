#!/bin/bash
set -ev

if [ "$INSTALL_TYPE" == "conda" ]; then
    source activate venv
else
    source venv/bin/activate
fi

PIPI="pip install --timeout=60 --find-links=$EXTRA_WHEELS"

if [ -n "$USE_PRE" ]; then
    PIPI="$PIPI --find-links=$PRE_WHEELS --pre";
fi

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
elif [ "$INSTALL_TYPE" == "conda" ]; then
    python setup.py install
fi

if [ "$TEST_WITH_XVFB" == "1" ]; then
    export DISPLAY=:99
fi
# -------------- Run the tests -----------------

# Change into an innocuous directory and find tests from installation
mkdir for_testing
cd for_testing
# We need the setup.cfg for the pytest settings
cp ../setup.cfg .
# No figure windows for mpl; quote to hide : from travis-ci yaml parsing
echo "backend : agg" > matplotlibrc
if [ "$COVERAGE" == "1" ]; then
    cp ../.coveragerc .;
    cp ../.codecov.yml .;
    # Run the tests and check for test coverage.
    coverage run -m pytest -svv --doctest-modules --verbose --durations=10 --pyargs dipy
    coverage report -m  # Generate test coverage report.
    codecov    # Upload the report to codecov.
else
    pytest -svv --doctest-modules --verbose --durations=10 --pyargs dipy
fi

