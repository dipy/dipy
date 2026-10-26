.. _data:

====
Data
====

DIPY provides access to several datasets used for testing, tutorials, and research. All datasets are downloaded automatically from the internet and cached locally on your machine for future use.

--------------------
How to get data
--------------------

The list of datasets can be retrieved using::

    from dipy.workflows.io import FetchFlow

    available_data = FetchFlow.get_fetcher_datanames().keys()


To retrieve all datasets, the following workflow can be run:

.. code-block:: python

    from tempfile import TemporaryDirectory

    from dipy.workflows.io import FetchFlow

    fetch_flow = FetchFlow()

    with TemporaryDirectory() as out_dir:
        fetch_flow.run(['all'], out_dir=out_dir)

If you want to download a particular dataset, you can do:

.. code-block:: python

    from tempfile import TemporaryDirectory

    from dipy.workflows.io import FetchFlow

    fetch_flow = FetchFlow()

    with TemporaryDirectory() as out_dir:
        fetch_flow.run(['bundle_fa_hcp'], out_dir=out_dir)

or::

    from dipy.data import fetch_bundle_fa_hcp

    files, folder = fetch_bundle_fa_hcp()


-------------
Datasets List
-------------

Details about datasets available in DIPY are described in the table below:

.. include:: dataset_list.rst
