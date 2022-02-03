#!/bin/bash
set -ev

if [ -e venv/bin/activate ]; then
    source venv/bin/activate
elif [ -e venv/Scripts/activate ]; then
    source virtenv/Scripts/activate
elif [ "$INSTALL_TYPE" == "conda" ]; then
    source activate venv
else
    echo Cannot activate virtual environment
    ls -R venv
    false
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
    echo "START UPLOAD COVERAGE"
    codecov    # Upload the report to codecov.
else
    pytest -svv --doctest-modules --verbose --durations=10 --pyargs dipy
fi
