.. _data:

====
Data
====

--------------------
How to get data
--------------------

The list of datasets can be retrieved using::

    from dipy.workflows.io import FetchFlow

    available_data = FetchFlow.get_fetcher_datanames().keys()


To retrieve all datasets, the following workflow can be run::

    from dipy.workflows.io import FetchFlow

    fetch_flow = FetchFlow()

    with TemporaryDirectory() as out_dir:
        fetch_flow.run(['all'])

If you want to download a particular dataset, you can do::

    from dipy.workflows.io import FetchFlow

    fetch_flow = FetchFlow()

    with TemporaryDirectory() as out_dir:
        fetch_flow.run(['bundle_fa_hcp'])

or::

    from dipy.data import fetch_bundle_fa_hcp

    files, folder = fetch_bundle_fa_hcp()


-------------
Datasets List
-------------

Details about datasets available in DIPY are described in the table below:

.. include:: dataset_list.rst

