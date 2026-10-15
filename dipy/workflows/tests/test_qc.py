from functools import partial
import html
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import socketserver
from tempfile import TemporaryDirectory
from threading import Thread
from urllib.error import HTTPError
from urllib.parse import unquote, urlencode
from urllib.request import Request, urlopen
import warnings
import zipfile

import numpy as np
import numpy.testing as npt
import trx.trx_file_memmap as tmm

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.reconst.dti import TensorModel, tensor_prediction
from dipy.workflows import qc, report
from dipy.workflows.cli import cli_flows
from dipy.workflows.combined_workflow import (
    _ReportHandler as ReportHandler,
    execute_semantic_pipeline,
)


def _flow_class(cli):
    module, name = cli_flows[cli]
    return getattr(importlib.import_module(module), name)


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
        assert len(result["figures"]) == (2 if qc.has_mpl else 0)

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

        page = (reports_dir / "01_gibbs.html").read_text(encoding="utf-8")
        sources = re.findall(r'src="([^"]*)"', page)
        assert bool(sources) is bool(qc.has_mpl)
        assert all(src.startswith("data:image/png;base64,") for src in sources)

        final = Path(report_path).read_text(encoding="utf-8")
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
        assert f"qcOpen({html.escape(json.dumps(volume), quote=True)})" in items
        assert 'href="affine.txt"' in items
        assert items.count("qc-open") == 2


def test_output_buttons_quote_paths(tmp_path):
    windows = os.name == "nt"
    # Windows forbids \ < > ? in file names, its paths still hold backslashes
    folder = tmp_path / ("o'brien & y" if windows else "o'brien \\ <x> & y")
    folder.mkdir()
    volume = str(folder / "fa.nii.gz")
    save_nifti(volume, np.zeros((4, 4, 4), dtype=np.float32), np.eye(4))
    text = str(folder / ("a b#c.txt" if windows else "a b#c?.txt"))
    np.savetxt(text, np.eye(4))

    items = report._output_items(
        outputs={"out_fa": volume, "out_affine": text}, page_dir=str(tmp_path)
    )
    (onclick,) = re.findall(r'onclick="qcOpen\(([^"]*)\)"', items)
    assert json.loads(html.unescape(onclick)) == volume
    (href,) = re.findall(r'class="qc-open" href="([^"]*)"', items)
    assert (
        unquote(html.unescape(href)) == Path(os.path.relpath(text, tmp_path)).as_posix()
    )
    assert "#" not in href and "?" not in href and " " not in href
    assert "<x>" not in items


def _serve(*, out_dir, token):
    """Start a report server on a free port; return it and its base URL."""
    handler = partial(ReportHandler, directory=out_dir, token=token)
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _post(url, **fields):
    return urlopen(Request(url, data=urlencode(fields).encode()))


def test_report_server_refuses_foreign_path():
    with TemporaryDirectory() as out_dir:
        httpd, base = _serve(out_dir=out_dir, token="secret")
        try:
            with npt.assert_raises(HTTPError) as cm:
                _post(f"{base}/qc-open", path="/etc/hosts", token="secret")
            assert cm.exception.code == 403
            assert "not a file served by this report" in cm.exception.read().decode()
        finally:
            httpd.shutdown()
            httpd.server_close()


def test_report_server_requires_its_token():
    with TemporaryDirectory() as out_dir:
        page = os.path.join(out_dir, "report.html")
        with open(page, "w", encoding="utf-8") as f:
            f.write("<html><head><title>r</title></head><body></body></html>")
        volume = os.path.join(out_dir, "fa.nii.gz")
        save_nifti(volume, np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))

        for token in ("secret", None):
            httpd, base = _serve(out_dir=out_dir, token=token)
            try:
                served = urlopen(f"{base}/report.html").read().decode()
                meta = f'<meta name="dipy-qc-token" content="{token}">'
                assert (meta in served) is (token is not None)
                assert "<title>r</title>" in served
                for request, code in (
                    (lambda base=base: urlopen(f"{base}/qc-open?path={volume}"), 405),
                    (lambda base=base: _post(f"{base}/qc-open", path=volume), 403),
                    (
                        lambda base=base: _post(
                            f"{base}/qc-open", path=volume, token="x"
                        ),
                        403,
                    ),
                    (
                        lambda base=base: _post(
                            f"{base}/qc-open", path=volume, token=""
                        ),
                        403,
                    ),
                ):
                    with npt.assert_raises(HTTPError) as cm:
                        request()
                    assert cm.exception.code == code
            finally:
                httpd.shutdown()
                httpd.server_close()

        with open(page, encoding="utf-8") as f:
            assert "dipy-qc-token" not in f.read()


PENDING = 'class="stage-badge badge-pending"'


def _reject_constant(constant):
    raise AssertionError(f"non-standard JSON constant {constant}")


def _stage_record(*, name, cli, status, duration=1.0):
    return {
        "name": name,
        "cli": cli,
        "duration": duration,
        "outputs": {},
        "qc": qc.qc_result(status=status),
    }


def _final_report(*, out_dir, config, stage_names, run_id=None, config_file=None):
    return Path(
        report.generate_html_report(
            config=config,
            config_file_path=config_file or str(out_dir / "config.toml"),
            execution_info={"execution_order": stage_names, "stages": []},
            output_path=str(out_dir / "pipeline_report.html"),
            reports_dir=str(out_dir / "reports"),
            run_id=run_id,
        )
    ).read_text(encoding="utf-8")


def test_report_escapes_user_strings(tmp_path):
    evil = "<script>alert(1)</script>"
    names = ["denoise"]
    report.write_stage_report(
        reports_dir=str(tmp_path / "reports"),
        stage_names=names,
        index=1,
        record=_stage_record(name="denoise", cli=f"dipy_x{evil}", status="PASS"),
        pipeline_name=evil,
    )
    config = {
        "General": {"name": evil, "description": evil, "author": evil},
        "io": {"out_dir": evil, "dwi": evil, "bvals": evil, "bvecs": evil},
    }
    page = _final_report(
        out_dir=tmp_path,
        config=config,
        stage_names=names,
        config_file=str(tmp_path / f"{evil}.toml"),
    )
    stage_page = (tmp_path / "reports" / "01_denoise.html").read_text(encoding="utf-8")
    for text in (page, stage_page):
        assert evil not in text
        assert html.escape(evil) in text


def test_report_links_are_relative_urls(tmp_path):
    conf_dir = tmp_path / "my conf"
    conf_dir.mkdir()
    config_file = conf_dir / "pipe#1.toml"
    config_file.write_text("", encoding="utf-8")
    out_dir = tmp_path / "out"
    names = ["denoise"]
    report.write_stage_report(
        reports_dir=str(out_dir / "reports"),
        stage_names=names,
        index=1,
        record=_stage_record(name="denoise", cli="dipy_denoise", status="PASS"),
        pipeline_name="p",
    )
    page = _final_report(
        out_dir=out_dir,
        config={"General": {"name": "p"}},
        stage_names=names,
        config_file=str(config_file),
    )
    assert 'href="../my%20conf/pipe%231.toml"' in page
    assert 'href="reports/01_denoise.html"' in page

    batch = tmp_path / "batch"
    run_dir = batch / "sub 01"
    (run_dir / "reports").mkdir(parents=True)
    (run_dir / "reports" / "index.html").write_text("", encoding="utf-8")
    path = report.generate_batch_report(
        results=[{"label": "r", "out_dir": str(run_dir), "status": "completed"}],
        output_path=str(batch / "summary.html"),
        pipeline_name="p",
        bids_dir=str(tmp_path),
    )
    assert (
        Path(path)
        .read_text(encoding="utf-8")
        .count('href="sub%2001/reports/index.html"')
        == 2
    )


def test_report_ignores_records_of_other_runs(tmp_path):
    reports_dir = str(tmp_path / "reports")
    names = ["denoise", "brain_mask"]

    def write(*, index, status, run_id):
        report.write_stage_report(
            reports_dir=reports_dir,
            stage_names=names,
            index=index,
            record=_stage_record(name=names[index - 1], cli="dipy_x", status=status),
            pipeline_name="p",
            run_id=run_id,
        )

    write(index=1, status="PASS", run_id="first")
    write(index=2, status="PASS", run_id="first")
    write(index=1, status="FAIL", run_id="second")

    records = report.load_stage_records(
        reports_dir=reports_dir, stage_names=names, run_id="second"
    )
    assert records[0]["qc"]["status"] == "FAIL"
    assert records[0]["run_id"] == "second"
    assert records[1] is None
    assert all(report.load_stage_records(reports_dir=reports_dir, stage_names=names))

    page = _final_report(
        out_dir=tmp_path, config={}, stage_names=names, run_id="second"
    )
    assert PENDING in page
    assert 'id="02_brain_mask"' not in page
    index = Path(
        report.write_index(
            reports_dir=reports_dir,
            stage_names=names,
            pipeline_name="p",
            run_id="second",
        )
    ).read_text(encoding="utf-8")
    assert PENDING in index

    page = _final_report(out_dir=tmp_path, config={}, stage_names=names)
    assert PENDING not in page
    assert 'id="02_brain_mask"' in page


def test_non_finite_values_stay_out_of_json(tmp_path):
    before = np.zeros((16, 16, 16, 2), dtype=np.float32)
    before[4:12, 4:12, 4:12] = np.random.default_rng(0).uniform(
        50, 100, size=(8, 8, 8, 2)
    )
    after = before.copy()
    after[4:12, 4:12, 4:12] *= 0.9
    paths = {}
    for name, data in (("before", before), ("after", after)):
        paths[name] = str(tmp_path / f"{name}.nii.gz")
        save_nifti(paths[name], data, np.eye(4))
    bvals = str(tmp_path / "dwi.bval")
    np.savetxt(bvals, [[0, 0]], fmt="%d")

    result = qc.run_stage_qc(
        cli="dipy_denoise",
        outputs={"out_denoised": paths["after"]},
        inputs={"input_files": paths["before"], "bvalues_files": bvals},
        io_config={},
    )
    snrs = {m["name"]: m for m in result["metrics"] if m["name"].startswith("SNR")}
    assert [m["status"] for m in snrs.values()] == ["INFO", "INFO"]
    assert [m["value"] for m in snrs.values()] == [None, None]
    json.dumps(result, allow_nan=False)

    record = {**_stage_record(name="denoise", cli="x", status="PASS"), "qc": result}
    record["duration"] = math.nan
    page = report.write_stage_report(
        reports_dir=str(tmp_path / "reports"),
        stage_names=["denoise"],
        index=1,
        record=record,
        pipeline_name="p",
    )
    text = (tmp_path / "reports" / "01_denoise.json").read_text(encoding="utf-8")
    json.loads(text, parse_constant=_reject_constant)
    assert "N/A" in Path(page).read_text(encoding="utf-8")


def _dti_phantom(out_dir):
    """Phantom plus brain mask and DTI eigenvalues/eigenvectors of a WLS fit."""
    paths = _phantom(out_dir)
    data, affine = load_nifti(paths["dwi"])
    bvals, bvecs = read_bvals_bvecs(paths["bvals"], paths["bvecs"])
    mask = data[..., 0] > 500
    fit = TensorModel(gradient_table(bvals, bvecs=bvecs)).fit(data, mask=mask)
    for name, volume in (
        ("mask", mask.astype(np.uint8)),
        ("evals", fit.evals.astype(np.float32)),
        ("evecs", fit.evecs.astype(np.float32)),
    ):
        paths[name] = str(Path(out_dir) / f"{name}.nii.gz")
        save_nifti(paths[name], volume, affine)
    return paths


def _metric_value(result, name):
    return next(m["value"] for m in result["metrics"] if m["name"] == name)


def test_dti_qc_uses_stage_gradients(tmp_path):
    paths = _dti_phantom(tmp_path)
    wrong_bvecs = str(tmp_path / "wrong.bvec")
    bvecs = np.loadtxt(paths["bvecs"])
    np.savetxt(wrong_bvecs, bvecs[[1, 2, 0]], fmt="%.6f")
    outputs = {"out_eval": paths["evals"], "out_evec": paths["evecs"]}
    inputs = {"input_files": paths["dwi"], "mask_files": paths["mask"]}
    name = "Tensor fit residual (median NRMSE)"

    stage = qc.run_stage_qc(
        cli="dipy_fit_dti",
        outputs=outputs,
        inputs={
            **inputs,
            "bvalues_files": paths["bvals"],
            "bvectors_files": [paths["bvecs"]],
            "b0_threshold": 50,
        },
        io_config={"bvals": paths["bvals"], "bvecs": wrong_bvecs},
    )
    io_only = qc.run_stage_qc(
        cli="dipy_fit_dti",
        outputs=outputs,
        inputs=inputs,
        io_config={"bvals": paths["bvals"], "bvecs": wrong_bvecs},
    )
    assert _statuses(stage)[name] == "PASS"
    assert _metric_value(stage, name) < _metric_value(io_only, name)


def test_dti_residual_matches_full_volume_computation(tmp_path):
    paths = _dti_phantom(tmp_path)
    result = qc.run_stage_qc(
        cli="dipy_fit_dti",
        outputs={"out_eval": paths["evals"], "out_evec": paths["evecs"]},
        inputs={"input_files": paths["dwi"], "mask_files": paths["mask"]},
        io_config={"bvals": paths["bvals"], "bvecs": paths["bvecs"]},
    )

    data = load_nifti(paths["dwi"])[0].astype(np.float32)
    mask = load_nifti(paths["mask"])[0] > 0
    evals = load_nifti(paths["evals"])[0].astype(np.float32)
    evecs = load_nifti(paths["evecs"])[0].astype(np.float32)
    bvals, bvecs = read_bvals_bvecs(paths["bvals"], paths["bvecs"])
    b0 = np.flatnonzero(bvals <= 50)
    s0 = data[..., b0].mean(-1)
    params = np.concatenate([evals, evecs.reshape(evecs.shape[:3] + (9,))], axis=-1)
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    predicted = np.zeros_like(data)
    predicted[mask] = tensor_prediction(params[mask], gtab, s0[mask])
    measured, model = data[mask], predicted[mask]
    measured = measured / np.maximum(measured[:, b0].mean(-1, keepdims=True), 1e-6)
    model = model / np.maximum(model[:, b0].mean(-1, keepdims=True), 1e-6)
    nrmse = np.sqrt(np.mean((model - measured) ** 2, axis=-1)) / np.maximum(
        measured.mean(axis=-1), 1e-6
    )
    nrmse_map = np.zeros(mask.shape, dtype=np.float32)
    nrmse_map[mask] = nrmse
    expected = float(np.nanmedian(nrmse_map[mask]))

    npt.assert_equal(
        _metric_value(result, "Tensor fit residual (median NRMSE)"), expected
    )


def test_foreground_mask_of_a_constant_volume():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for value in (0.0, 3.5):
            mask = qc.foreground_mask(volume=np.full((4, 4, 4), value))
            npt.assert_equal(mask.any(), False)
    volume = np.zeros((4, 4, 4))
    volume[1:3, 1:3, 1:3] = 10
    npt.assert_equal(qc.foreground_mask(volume=volume), volume > 0)


def test_force_qc_uses_stage_bvals(tmp_path):
    paths = _dti_phantom(tmp_path)
    result = qc.run_stage_qc(
        cli="dipy_fit_force",
        outputs={"out_predicted_signal": paths["dwi"]},
        inputs={
            "input_files": paths["dwi"],
            "mask_files": paths["mask"],
            "bvalues_files": paths["bvals"],
        },
        io_config={},
    )
    assert result["status"] != "ERROR", result["error"]
    npt.assert_equal(_metric_value(result, "Signal matching error (median NRMSE)"), 0.0)


def test_median_otsu_gets_brain_mask_qc(tmp_path):
    paths = _dti_phantom(tmp_path)
    result = qc.run_stage_qc(
        cli="dipy_median_otsu",
        outputs={"out_mask": paths["mask"], "out_masked": paths["dwi"]},
        inputs={"input_files": paths["dwi"], "bvalues_files": [paths["bvals"]]},
        io_config={},
    )
    assert result["status"] != "ERROR", result["error"]
    assert _statuses(result)["Connected components"] == "PASS"


def test_b0_extraction_qc_on_3d_input(tmp_path):
    volume = np.zeros((16, 16, 16), dtype=np.float32)
    volume[4:12, 4:12, 4:12] = np.linspace(100, 200, 8)[None, None, :]
    path = str(tmp_path / "b0.nii.gz")
    save_nifti(path, volume, np.eye(4))
    bvals = str(tmp_path / "b0.bval")
    np.savetxt(bvals, [[0]], fmt="%d")
    result = qc.run_stage_qc(
        cli="dipy_extract_b0",
        outputs={},
        inputs={"input_files": path},
        io_config={"bvals": bvals},
    )
    npt.assert_almost_equal(_metric_value(result, "CV of b0 mean intensity"), 0)


def _extracted_dir(streamlines):
    """Folder a compressed ``.trx`` was extracted to, from its memmap."""
    return Path(streamlines._data.filename).parent


def test_trx_tractograms_are_closed(tmp_path):
    ref = str(tmp_path / "ref.nii.gz")
    save_nifti(ref, np.zeros((40, 40, 40), dtype=np.float32), np.eye(4))
    line = np.linspace(0, 30, 31)
    streamlines = [np.c_[line, np.full(31, y), np.full(31, 5.0)] for y in range(12)]
    sft = StatefulTractogram(streamlines, ref, Space.RASMM)
    path = str(tmp_path / "tracts.trx")
    tmm.save(tmm.TrxFile.from_sft(sft), path, compression_standard=zipfile.ZIP_DEFLATED)

    with qc._open_streamlines(path=path) as loaded:
        assert len(loaded) == 12
        extracted = _extracted_dir(loaded)
        assert extracted.name.startswith("trx_")
        assert extracted.is_dir()
    assert not extracted.exists()

    result = qc.run_stage_qc(
        cli="dipy_track", outputs={"out_tractogram": path}, inputs={}, io_config={}
    )
    assert _statuses(result)["Streamlines"] == "PASS"


def test_qc_reads_files_with_glob_characters(tmp_path):
    folder = tmp_path / ("sub[01]" if os.name == "nt" else "sub[01]*?")
    folder.mkdir()
    paths = _dti_phantom(folder)
    result = qc.run_stage_qc(
        cli="dipy_brain_mask",
        outputs={"out_mask": paths["mask"]},
        inputs={"input_files": paths["dwi"], "bvalues_files": paths["bvals"]},
        io_config={},
    )
    assert result["status"] != "ERROR", result["error"]
    assert _statuses(result)["Connected components"] == "PASS"


def test_streamlines_of_every_format_are_in_rasmm(tmp_path):
    affine = np.array(
        [[2.0, 0, 0, -30], [0, 1.5, 0, -20], [0, 0, 2.5, -10], [0, 0, 0, 1]]
    )
    ref = str(tmp_path / "ref.nii.gz")
    save_nifti(ref, np.zeros((40, 40, 40), dtype=np.float32), affine)
    line = np.linspace(0, 30, 31)
    streamlines = [np.c_[line, np.full(31, y), np.full(31, 5.0)] for y in range(12)]
    streamlines += [np.c_[np.full(4, 5.0), np.full(4, 5.0), [0, 1, 2, 3.0]]] * 3

    results = {}
    for ext in ("trk", "tck", "trx"):
        path = _save_streamlines(
            path=tmp_path / f"tracts.{ext}", streamlines=streamlines, reference=ref
        )
        with qc._open_streamlines(path=path) as loaded:
            npt.assert_allclose(
                np.concatenate(list(loaded)), np.concatenate(streamlines), atol=1e-4
            )
        result = qc.run_stage_qc(
            cli="dipy_track", outputs={"out_tractogram": path}, inputs={}, io_config={}
        )
        assert result["status"] != "ERROR", result["error"]
        results[ext] = {m["name"]: m["value"] for m in result["metrics"]}
    assert results["tck"] == results["trk"] == results["trx"]


def _b0_last_pair(tmp_path):
    """Before/after DWI pair whose only b0 is the last volume.

    The b0 is identical in both files while the first, diffusion-weighted
    volume is not, so a QC that misses the b-values and falls back on
    volume 0 sees a difference that the b0 does not have.
    """
    rng = np.random.default_rng(0)
    before = np.zeros((16, 16, 16, 2), dtype=np.float32)
    before[4:12, 4:12, 4:12, 0] = rng.uniform(50, 100, size=(8, 8, 8))
    before[4:12, 4:12, 4:12, 1] = 100
    after = before.copy()
    after[4:12, 4:12, 4:12, 0] = rng.uniform(50, 100, size=(8, 8, 8))
    paths = {}
    for name, data in (("before", before), ("after", after)):
        paths[name] = str(tmp_path / f"{name}.nii.gz")
        save_nifti(paths[name], data, np.eye(4))
    paths["bvals"] = str(tmp_path / "dwi.bval")
    np.savetxt(paths["bvals"], [[1000, 0]], fmt="%d")
    return paths


def test_qc_reads_workflow_specific_bval_names(tmp_path):
    paths = _b0_last_pair(tmp_path)
    result = qc.run_stage_qc(
        cli="dipy_denoise_patch2self",
        outputs={"out_denoised": paths["after"]},
        inputs={"input_files": paths["before"], "bval_files": paths["bvals"]},
        io_config={},
    )
    npt.assert_almost_equal(_metric_value(result, "SSIM before/after (mean b0)"), 1)

    result = qc.run_stage_qc(
        cli="dipy_correct_biasfield",
        outputs={"out_corrected": paths["after"]},
        inputs={"input_files": paths["before"], "bval": paths["bvals"]},
        io_config={},
    )
    npt.assert_almost_equal(
        _metric_value(result, "Intensity CV before (foreground)"), 0
    )


def test_qc_gradient_names_match_workflows():
    params = {
        cli: set(inspect.signature(_flow_class(cli).run).parameters)
        for cli in qc.QC_FUNCS
    }
    for cli, names in params.items():
        bvals = [p for p in names if p.startswith("bval")]
        bvecs = [p for p in names if p.startswith("bvec") and p != "bvecs_tol"]
        assert set(bvals) <= set(qc.BVAL_PARAMS), cli
        assert set(bvecs) <= set(qc.BVEC_PARAMS), cli


def test_bundle_names_with_glob_characters_in_atlas_dir(tmp_path):
    ref = str(tmp_path / "ref.nii.gz")
    save_nifti(ref, np.zeros((40, 40, 40), dtype=np.float32), np.eye(4))
    line = np.linspace(0, 30, 31)
    bundle = [np.c_[line, np.full(31, y), np.full(31, 5.0)] for y in range(12)]
    atlas_dir = tmp_path / "atlas[v1]"
    atlas_dir.mkdir()
    (atlas_dir / "AF_L.trk").touch()
    rec_dir = tmp_path / "rec_bundles"
    rec_dir.mkdir()
    _save_streamlines(
        path=rec_dir / "moved_AF_L__recognized.trx", streamlines=bundle, reference=ref
    )

    result = qc.run_stage_qc(
        cli="dipy_recobundles",
        outputs={"out_dir": str(rec_dir)},
        inputs={},
        io_config={"bundle_atlas_dir": str(atlas_dir)},
    )
    assert _statuses(result) == {"AF_L (mean length 30 mm)": "PASS"}

    result = qc.run_stage_qc(
        cli="dipy_recobundles",
        outputs={"out_dir": str(rec_dir)},
        inputs={},
        io_config={},
    )
    assert _statuses(result) == {"moved_AF_L (mean length 30 mm)": "PASS"}


def test_truncated_stage_record_is_pending(tmp_path):
    reports_dir = tmp_path / "reports"
    names = ["denoise", "brain_mask"]
    for index, name in enumerate(names, start=1):
        report.write_stage_report(
            reports_dir=str(reports_dir),
            stage_names=names,
            index=index,
            record=_stage_record(name=name, cli="dipy_x", status="PASS"),
            pipeline_name="p",
        )
    record = reports_dir / "02_brain_mask.json"
    record.write_text(record.read_text()[:20], encoding="utf-8")
    assert not list(reports_dir.glob("*.tmp"))

    records = report.load_stage_records(reports_dir=str(reports_dir), stage_names=names)
    assert records[0] is not None and records[1] is None
    page = _final_report(out_dir=tmp_path, config={}, stage_names=names)
    assert PENDING in page
