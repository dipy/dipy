#!/bin/bash
set -ev

if [ "$INSTALL_TYPE" == "conda" ]; then
    source activate venv
else
    source venv/bin/activate
fi

PIPI="pip install --timeout=60"

if [ "$USE_PRE" == "1" ]; then
    PIPI="$PIPI --extra-index-url=$PRE_WHEELS --pre";
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
