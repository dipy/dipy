#!/bin/bash
set -ev

if [ "$INSTALL_TYPE" == "conda" ]; then

    conda config --set always_yes yes --set changeps1 no
    if [ "$AGENT_OS" == "Linux" ]; then
        # Workaround: https://github.com/conda/conda/issues/9337
        pip uninstall -y setuptools
        conda install -yq setuptools
    fi
    conda update -yq conda
    conda install conda-build anaconda-client
    conda config --add channels conda-forge
    conda create -n venv --yes python=$PYTHON_VERSION pip
    conda install -yq --name venv $DEPENDS $EXTRA_DEPENDS pytest
else
    PIPI="pip install --timeout=60 --find-links=$EXTRA_WHEELS"

    if [ -n "$USE_PRE" ]; then
        PIPI="$PIPI --find-links=$PRE_WHEELS --pre";
    fi

    pip install --upgrade virtualenv
    virtualenv $VENV_ARGS venv
    source venv/bin/activate
    # just to check python version
    python --version

    if [ "$AGENT_OS" == "Linux" ]; then
        # Needed for Python 3.5 wheel fetching
        $PIPI --upgrade pip setuptools
    fi

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
