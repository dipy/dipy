#!/bin/bash

echo "Activate virtual environment"
source tools/ci/activate_env.sh

set -ex

echo "Display Python version"
python -c "import sys; print(sys.version)"
python -m pip install -U pip setuptools>=30.3.0 wheel


echo "Install Dependencies"
if [ "$INSTALL_TYPE" == "conda" ]; then
    conda install -yq --name venv $DEPENDS $EXTRA_DEPENDS pytest
else
    PIPI="pip install --timeout=60 "

    if [ "$USE_PRE" == "1" ] || [ "$USE_PRE" == true ]; then
        # --index-url takes priority over --extra-index-url, so that packages, and
        # their dependencies, with versions available in the nightly channel will be installed before falling back to the Python Package Index.
        PIPI="$PIPI --pre --index-url $PRE_WHEELS --extra-index-url https://pypi.org/simple";
    fi

    $PIPI pytest==8.0.0
    $PIPI numpy
    if [ -n "$DEPENDS" ]; then $PIPI $DEPENDS $EXTRA_DEPENDS; fi
    if [ "$COVERAGE" == "1" ] || [ "$COVERAGE" = true ]; then pip install coverage coveralls; fi
fi

set +ex
