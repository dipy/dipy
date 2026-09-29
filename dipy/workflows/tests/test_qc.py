from functools import partial
from pathlib import Path
import re
import socketserver
from tempfile import TemporaryDirectory
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import numpy.testing as npt

from dipy.io.image import save_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.workflows import qc, report
from dipy.workflows.combined_workflow import (
    _ReportHandler as ReportHandler,
    execute_semantic_pipeline,
)


def _statuses(result):
    return {m["name"]: m["status"] for m in result["metrics"]}


def _phantom(out_dir):
    """Write a noisy spherical DWI phantom with 2 b0 and 12 b=1000 volumes."""
    rng = np.random.default_rng(42)
    shape = (24, 24, 16)
    grid = np.indices(shape).astype(float)
    center = (np.array(shape) - 1)[:, None, None, None] / 2
    brain = np.sqrt((((grid - center) / [[[[9]]], [[[9]]], [[[6]]]]) ** 2).sum(0)) < 1

    bvecs = rng.normal(size=(12, 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    bvecs = np.vstack([np.zeros((2, 3)), bvecs])
    bvals = np.array([0, 0] + [1000] * 12, dtype=float)
    tensor = np.diag([1.5e-3, 0.4e-3, 0.4e-3])
    adc = np.einsum("ij,jk,ik->i", bvecs, tensor, bvecs)
    signal = 1000 * np.exp(-bvals * adc)
    data = brain[..., None] * signal + rng.normal(scale=20, size=shape + (14,))

    paths = {
        "dwi": str(Path(out_dir) / "dwi.nii.gz"),
        "bvals": str(Path(out_dir) / "dwi.bval"),
        "bvecs": str(Path(out_dir) / "dwi.bvec"),
    }
    save_nifti(paths["dwi"], np.abs(data).astype(np.float32), np.diag([2.5] * 3 + [1]))
    np.savetxt(paths["bvals"], bvals[None], fmt="%d")
    np.savetxt(paths["bvecs"], bvecs.T, fmt="%.6f")
    return paths


def test_ssim_3d():
    rng = np.random.default_rng(0)
    a = rng.random((20, 20, 20))
    npt.assert_almost_equal(qc.ssim_3d(a=a, b=a), 1.0)
    low = qc.ssim_3d(a=a, b=a + 0.1 * rng.random(a.shape))
    high = qc.ssim_3d(a=a, b=a + 0.5 * rng.random(a.shape))
    assert 1.0 > low > high


def test_radial_amplitude_spectrum():
    rng = np.random.default_rng(0)
    smooth = np.add.outer(
        np.add.outer(np.arange(16.0), np.arange(16.0)), np.arange(16.0)
    )
    freqs, smooth_amp = qc.radial_amplitude_spectrum(volume=smooth)
    _, noise_amp = qc.radial_amplitude_spectrum(volume=rng.random(smooth.shape))
    low = freqs < 0.1
    assert (
        smooth_amp[low].sum() / smooth_amp.sum()
        > noise_amp[low].sum() / noise_amp.sum()
    )


def test_brain_mask_qc():
    with TemporaryDirectory() as out_dir:
        paths = _phantom(out_dir)
        mask = np.zeros((24, 24, 16), dtype=np.uint8)
        mask[2:6, 2:6, 2:6] = 1
        mask[15:20, 15:20, 8:12] = 1
        mask_path = str(Path(out_dir) / "mask.nii.gz")
        save_nifti(mask_path, mask, np.diag([2.5] * 3 + [1]))

        result = qc.run_stage_qc(
            cli="dipy_brain_mask",
            outputs={"out_mask": mask_path},
            inputs={"input_files": paths["dwi"]},
            io_config=paths,
        )
        statuses = _statuses(result)
        assert statuses["Connected components"] == "WARN"
        assert statuses["Fraction of field of view"] == "PASS"
        assert result["status"] == "WARN"


def test_run_stage_qc_error():
    result = qc.run_stage_qc(cli="dipy_brain_mask", outputs={}, inputs={}, io_config={})
    assert result["status"] == "ERROR"
    assert "KeyError" in result["error"]


def test_generic_qc_flags_non_finite():
    with TemporaryDirectory() as out_dir:
        data = np.ones((8, 8, 8), dtype=np.float32)
        data[0, 0, 0] = np.nan
        path = str(Path(out_dir) / "map.nii.gz")
        save_nifti(path, data, np.eye(4))
        result = qc.run_stage_qc(
            cli="dipy_unknown", outputs={"out_map": path}, inputs={}, io_config={}
        )
        assert _statuses(result) == {"out_map: non-finite voxels": "WARN"}


def _save_streamlines(*, path, streamlines, reference):
    sft = StatefulTractogram(streamlines, reference, Space.RASMM)
    save_tractogram(sft, str(path), bbox_valid_check=False)
    return str(path)


def test_streamline_qc():
    with TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        ref = str(out_dir / "ref.nii.gz")
        save_nifti(ref, np.zeros((40, 40, 40), dtype=np.float32), np.eye(4))
        line = np.linspace(0, 30, 31)
        bundle_a = [np.c_[line, np.full(31, y), np.full(31, 5.0)] for y in range(12)]
        bundle_b = [np.c_[np.full(4, 5.0), np.full(4, 5.0), [0, 1, 2, 3.0]]] * 3

        tractogram = _save_streamlines(
            path=out_dir / "tracts.trx", streamlines=bundle_a + bundle_b, reference=ref
        )
        result = qc.run_stage_qc(
            cli="dipy_track",
            outputs={"out_tractogram": tractogram},
            inputs={},
            io_config={},
        )
        assert _statuses(result)["Streamlines"] == "PASS"
        assert _statuses(result)["Streamlines shorter than 10 mm"] == "PASS"
        assert len(result["figures"]) == 2

        affine = str(out_dir / "affine.txt")
        np.savetxt(affine, np.eye(4))
        result = qc.run_stage_qc(
            cli="dipy_slr",
            outputs={
                "out_affine": affine,
                "out_moved": tractogram,
                "out_stat_centroids": tractogram,
                "out_moving_centroids": tractogram,
                "out_moved_centroids": tractogram,
            },
            inputs={"static_files": tractogram},
            io_config={},
        )
        statuses = _statuses(result)
        assert statuses["Scale"] == "PASS"
        assert statuses["Centroid distance after"] == "PASS"

        atlas_dir = out_dir / "atlas"
        atlas_dir.mkdir()
        for name in ("AF_L", "CST"):
            (atlas_dir / f"{name}.trk").touch()
        rec_dir = out_dir / "rec_bundles"
        rec_dir.mkdir()
        first = _save_streamlines(
            path=rec_dir / "moved_AF_L__recognized.trx",
            streamlines=bundle_a,
            reference=ref,
        )
        _save_streamlines(
            path=rec_dir / "moved_CST__recognized.trx",
            streamlines=bundle_b,
            reference=ref,
        )
        result = qc.run_stage_qc(
            cli="dipy_recobundles",
            outputs={"out_dir": str(rec_dir), "out_recognized_transf": first},
            inputs={},
            io_config={"bundle_atlas_dir": str(atlas_dir)},
        )
        statuses = _statuses(result)
        assert statuses["AF_L (mean length 30 mm)"] == "PASS"
        assert statuses["CST (mean length 3 mm)"] == "WARN"


def test_pipeline_writes_stage_reports():
    with TemporaryDirectory() as out_dir:
        io_config = {**_phantom(out_dir), "out_dir": out_dir}
        config = {
            "General": {"name": "qc_test"},
            "io": io_config,
            "pipeline": [
                {
                    "name": "gibbs",
                    "cli": "dipy_gibbs_ringing",
                    "input_files": "${io.dwi}",
                },
                {
                    "name": "reslice",
                    "cli": "dipy_reslice",
                    "input_files": "${gibbs.out_unring}",
                    "new_vox_size": [2.0],
                },
                {
                    "name": "b0_extraction",
                    "cli": "dipy_extract_b0",
                    "input_files": "${reslice.out_resliced}",
                    "bvalues_files": "${io.bvals}",
                },
                {
                    "name": "brain_mask",
                    "cli": "dipy_brain_mask",
                    "input_files": "${reslice.out_resliced}",
                    "bvalues_files": "${io.bvals}",
                    "method": "median_otsu",
                },
                {
                    "name": "dti_fit",
                    "cli": "dipy_fit_dti",
                    "input_files": "${reslice.out_resliced}",
                    "bvalues_files": "${io.bvals}",
                    "bvectors_files": "${io.bvecs}",
                    "mask_files": "${brain_mask.out_mask}",
                    "out_dir": f"{out_dir}/dti",
                },
            ],
        }
        report_path = str(Path(out_dir) / "pipeline_report.html")
        execute_semantic_pipeline(
            config=config, io_config=io_config, report_path=report_path
        )

        reports_dir = Path(out_dir) / "reports"
        names = [s["name"] for s in config["pipeline"]]
        records = report.load_stage_records(
            reports_dir=str(reports_dir), stage_names=names
        )
        for index, (name, record) in enumerate(zip(names, records), start=1):
            base = report.stage_basename(index=index, stage_name=name)
            assert (reports_dir / f"{base}.html").exists()
            assert record["qc"]["status"] != "ERROR", record["qc"]["error"]
            assert record["qc"]["metrics"]
        assert (reports_dir / "index.html").exists()

        page = (reports_dir / "01_gibbs.html").read_text()
        sources = re.findall(r'src="([^"]*)"', page)
        assert sources
        assert all(src.startswith("data:image/png;base64,") for src in sources)

        final = Path(report_path).read_text()
        for index, name in enumerate(names, start=1):
            assert (
                f'id="{report.stage_basename(index=index, stage_name=name)}"' in final
            )
        assert "reports/03_b0_extraction.html" in final


def test_output_buttons():
    with TemporaryDirectory() as out_dir:
        volume = str(Path(out_dir) / "fa.nii.gz")
        save_nifti(volume, np.zeros((4, 4, 4), dtype=np.float32), np.eye(4))
        text = str(Path(out_dir) / "affine.txt")
        np.savetxt(text, np.eye(4))

        items = report._output_items(
            outputs={"out_fa": volume, "out_affine": text, "out_dir": out_dir},
            page_dir=out_dir,
        )
        assert f"qcOpen('{volume}')" in items
        assert 'href="affine.txt"' in items
        assert items.count("qc-open") == 2


def test_report_server_refuses_foreign_path():
    with TemporaryDirectory() as out_dir:
        handler = partial(ReportHandler, directory=out_dir)
        with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
            Thread(target=httpd.handle_request, daemon=True).start()
            url = f"http://127.0.0.1:{httpd.server_address[1]}/qc-open?path=/etc/hosts"
            with npt.assert_raises(HTTPError) as cm:
                urlopen(url)
            # 404 when Skyline is not installed, 403 when it is: either way the
            # file outside the served directory is never opened.
            assert cm.exception.code in (403, 404)
            assert cm.exception.read().decode()
