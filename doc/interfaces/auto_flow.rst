.. _auto_flow:

=====================
Automatic Pipelines
=====================

``dipy_auto`` runs a whole diffusion pipeline from a single command. A pipeline
is a list of stages described in a TOML file; each stage names a DIPY command
line workflow and how its inputs connect to the outputs of the stages before
it. Nine pipelines ship with DIPY, from plain denoising to a full
preprocessing, reconstruction, tracking and bundle segmentation chain.

List them with::

    dipy_auto --list_pipelines

-------------------
One set of images
-------------------

Pass the diffusion image and its gradient table; the file types are recognised
from their extensions::

    dipy_auto dwi.nii.gz dwi.bval dwi.bvec --pipeline_type preprocessing \
        --out_dir results

Add the companion images the pipeline needs::

    dipy_auto dwi.nii.gz dwi.bval dwi.bvec --t1_file t1.nii.gz \
        --mask_file brain_mask.nii.gz --out_dir results

``--dry_run`` prints the execution plan, the stage order and an estimate of the
disk space the run needs, without executing anything. ``--start <stage>``
resumes a pipeline from one stage, reusing the outputs already on disk.

Each run writes, into ``--out_dir``:

- ``<timestamp>_<pipeline>_pipeline.toml``, the exact configuration that ran;
- ``<timestamp>_dipy_auto.log``, the full log;
- ``<timestamp>_pipeline_report.html``, the final report;
- ``reports/NN_<stage>.html`` and ``reports/index.html``, the per-stage QC
  pages, refreshed as soon as each stage finishes.

Reopen a report, with its images served so the QC buttons work::

    dipy_auto results/20260929_120000_pipeline_report.html

-------------------------
A whole BIDS dataset
-------------------------

With ``--bids``, ``dipy_auto`` follows the BIDS-App argument convention and
runs the pipeline on every diffusion acquisition of a dataset::

    dipy_auto /data/ds001 /data/ds001/derivatives/dipy participant --bids \
        --pipeline_type preprocessing

The positional arguments are the dataset root, the output directory (optional)
and the analysis level (optional; only ``participant`` is implemented). When
the output directory is omitted, the results go to
``<bids_dir>/derivatives/dipy``. ``--bids_folder`` passes the dataset root as
an option instead, which is convenient together with a TOML configuration::

    dipy_auto my_pipeline.toml --bids --bids_folder /data/ds001

Restrict what runs with the entity filters, which accept labels with or
without their prefix::

    dipy_auto /data/ds001 --bids --participant_label 01 02 --session_label 1
    dipy_auto /data/ds001 --bids --acquisition_label b1000 --run_label 1

For every acquisition, ``dipy_auto`` resolves from the dataset:

- the b-value and b-vector files and the JSON sidecar, following the BIDS
  inheritance principle, so that metadata placed at the dataset, subject or
  session level applies to the files below it;
- the T1w image of the subject, preferring the same session;
- the EPI fieldmap acquired with the opposite phase encoding, by
  ``IntendedFor``, then by an opposite ``dir-`` entity, then by an opposite
  ``PhaseEncodingDirection``;
- ``PhaseEncodingDirection``, ``TotalReadoutTime``, ``EffectiveEchoSpacing``,
  ``EchoTime`` and ``RepetitionTime``, which are logged and stored in the
  ``[io]`` section of the generated configuration as ``${io.pe_dir}``,
  ``${io.total_readout_time}``, ``${io.effective_echo_spacing}``,
  ``${io.echo_time}`` and ``${io.repetition_time}``.

Everything that was inferred is reported in the log, so a run can be checked
against what the dataset actually contains.

Output layout
=============

The output is a BIDS-Derivatives dataset::

    derivatives/dipy/
        dataset_description.json
        <timestamp>_batch_summary.html
        <timestamp>_batch_summary.json
        <timestamp>_dipy_auto.log
        sub-01/
            dwi/
                <timestamp>_<pipeline>_pipeline.toml
                <timestamp>_pipeline_report.html
                reports/index.html
                dwi_denoised.nii.gz
                sub-01_desc-denoised_dwi.nii.gz
                sub-01_desc-denoised_dwi.json
                dti/
        sub-02/
            ses-1/
                dwi/
                    acq-b1000_dir-AP_run-1/
                    acq-b1000_dir-AP_run-2/

Every acquisition gets its own directory, so the per-stage QC reports and the
resume logic work per acquisition exactly as they do for a single subject. When
one session holds several acquisitions they would share a datatype directory
and overwrite each other, so each one is given a subdirectory named after the
entities that tell them apart.

Headline outputs — FA, MD, RD and AD maps, the denoised DWI, the brain mask and
tractograms — are additionally exposed at the datatype level under BIDS-style
names such as ``sub-01_desc-FA_dwimap.nii.gz``, each with a JSON sidecar
recording the stage and the source files. These are links to the stage outputs,
which are left where the pipeline wrote them.

The batch summary lists every acquisition with its status, duration, the stage
it failed at if it failed, and a link to its report. A failing acquisition does
not stop the batch — ``dipy_auto`` reports it, carries on, and exits non-zero
at the end. ``--stop_on_error`` stops at the first failure instead.

Validation
==========

``dipy_auto`` refuses to guess. A dataset root passed without ``--bids`` is an
error naming the command to use instead, and single-subject inputs such as a
positional DWI file or ``--t1_file`` cannot be combined with ``--bids``, since
they would be ambiguous over a whole dataset.

If `pybids <https://bids-standard.github.io/pybids/>`_ is installed, the file
names of the discovered acquisitions are checked against the BIDS validator and
any non-compliant name is reported as a warning. The discovery itself never
depends on pybids, so the BIDS mode works without it.
