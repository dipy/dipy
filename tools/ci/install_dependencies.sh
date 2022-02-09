#!/bin/bash
set -ev

echo "Activate virtual environment"
source tools/ci/activate.sh

echo "Display Python version"
python -c "import sys; print(sys.version)"
python -m pip install -U pip setuptools>=30.3.0 wheel


echo "Install Dependencies"
if [ "$INSTALL_TYPE" == "conda" ]; then
    conda install -yq --name venv $DEPENDS $EXTRA_DEPENDS pytest
else
    PIPI="pip install --timeout=60 "

    if [ "$USE_PRE" == "1" ]; then
        PIPI="$PIPI --extra-index-url=$PRE_WHEELS --pre";
    fi

    $PIPI pytest
    $PIPI numpy
    if [ -n "$DEPENDS" ]; then $PIPI $DEPENDS $EXTRA_DEPENDS; fi
    if [ "$COVERAGE" == "1" ]; then pip install coverage coveralls codecov; fi
fi


