#!/bin/bash

echo "Activate virtual environment"
source tools/ci/activate_env.sh

set -ex

PIPI="pip install --timeout=60 -Csetup-args=--vsenv -Ccompile-args=-v"

if [ "$USE_PRE" == "1" ] || [ "$USE_PRE" == true ]; then
    # --index-url takes priority over --extra-index-url, so that packages, and
    # their dependencies, with versions available in the nightly channel will be installed before falling back to the Python Package Index.
    PIPI="$PIPI --pre --index-url $PRE_WHEELS --extra-index-url https://pypi.org/simple";
fi

#---------- DIPY Installation -----------------

if [ "$INSTALL_TYPE" == "pip" ] || [ "$INSTALL_TYPE" == "conda" ]; then
    $PIPI -vv .
fi

set +ex