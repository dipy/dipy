#!/bin/bash
set -ev

if [ "$INSTALL_TYPE" == "conda" ]; then

    conda config --set always_yes yes --set changeps1 no
    # if [ "$AGENT_OS" == "Linux" ]; then
    # Workaround: https://github.com/conda/conda/issues/9337
    pip uninstall -y setuptools wheel
    conda install -yq setuptools wheel
    # fi
    conda update -yq conda
    conda install conda-build anaconda-client
    conda config --add channels conda-forge
    conda create -n venv --yes python=$PYTHON_VERSION pip
    conda install -yq --name venv $DEPENDS $EXTRA_DEPENDS pytest
else
    PIPI="pip install --timeout=60 "

    if [ "$USE_PRE" == "1" ]; then
        PIPI="$PIPI --extra-index-url=$PRE_WHEELS --pre";
    fi

    pip install --upgrade virtualenv
    virtualenv $VENV_ARGS venv
    source venv/bin/activate
    # just to check python version
    python --version

    # if [ "$AGENT_OS" == "Linux" ]; then
    $PIPI --upgrade pip setuptools wheel
    # fi

    $PIPI pytest
    $PIPI numpy
    if [ -n "$DEPENDS" ]; then $PIPI $DEPENDS $EXTRA_DEPENDS; fi
    if [ "$COVERAGE" == "1" ]; then pip install coverage coveralls codecov; fi
    if [ "$VTK" == "1" ]; then
        sudo apt-get update;
        sudo apt-get install -y $VTK_VER;
        sudo apt-get install -y xvfb;
        sudo apt-get install -y python-tk;
        sudo apt-get install -y python-imaging;
        $PIPI xvfbwrapper;
    fi
fi
