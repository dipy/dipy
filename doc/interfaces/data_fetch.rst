.. _data_fetch:

-------------------------
Downloading DIPY datasets
-------------------------
In this tutorial, you will learn how to download the DIPY datasets using the terminal.

First, let's create a folder::

    mkdir data_folder

You can list all available datasets in DIPY using the command::

    dipy_fetch list

In order to fetch a specific dataset, you will need to specify its name to the ``dipy_fetch``
command, and an optional destination directory using the ``--out_dir`` argument::

    dipy_fetch {specific_dataset} --out_dir {specific_data_out_folder}

For example, to download the ``sherbrooke_3shell`` dataset, you would run::

    dipy_fetch sherbrooke_3shell --out_dir data_folder

You can find details about all the datasets available in DIPY :ref:`data`