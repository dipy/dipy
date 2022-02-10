#!/bin/bash

echo "Activate virtual environment"
source tools/ci/activate_env.sh

set -ex

PIPI="pip install --timeout=60"

if [ "$USE_PRE" == "1" ] || [ "$USE_PRE" = true ]; then
    PIPI="$PIPI --extra-index-url=$PRE_WHEELS --pre";
fi

#---------- DIPY Installation -----------------

if [ "$INSTALL_TYPE" == "setup" ]; then
    python setup.py install
elif [ "$INSTALL_TYPE" == "pip" ]; then
    $PIPI .
elif [ "$INSTALL_TYPE" == "sdist" ]; then
    # python -m pep517.build
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
    $PIPI .
fi

set +ex