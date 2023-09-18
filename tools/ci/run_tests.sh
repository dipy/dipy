#!/bin/bash

echo "Activate virtual environment"
source tools/ci/activate_env.sh

set -ex

echo "Run the tests"

# Change into an innocuous directory and find tests from installation
mkdir for_testing
cd for_testing
# We need the setup.cfg for the pytest settings
cp ../pyproject.toml .
# No figure windows for mpl; quote to hide : from travis-ci yaml parsing
echo "backend : agg" > matplotlibrc
if [ "$COVERAGE" == "1" ] || [ "$COVERAGE" == true ]; then
    cp ../.coveragerc .;
    cp ../.codecov.yml .;
    # Run the tests and check for test coverage.
    coverage run -m pytest -c pyproject.toml -svv --doctest-modules --verbose --durations=10 --pyargs dipy
    coverage report -m  # Generate test coverage report.
    coverage xml  # Generate coverage report in xml format for codecov upload.
else
    pytest -c pyproject.toml -svv --doctest-modules --verbose --durations=10 --pyargs dipy
fi
cd ..
set +ex