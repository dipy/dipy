if [ -e venv/bin/activate ]; then
    source venv/bin/activate
elif [ -e venv/Scripts/activate ]; then
    source venv/Scripts/activate
elif [ "$INSTALL_TYPE" == "conda" ]; then
    conda init bash
    source $CONDA/etc/profile.d/conda.sh
    conda activate venv
else
    echo Cannot activate virtual environment
    ls -R venv
    false
fi