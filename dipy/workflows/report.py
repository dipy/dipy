"""HTML report generation for dipy_auto pipeline execution.

This module generates detailed HTML reports showing pipeline configuration,
execution graph, timing information, and slice mosaic visualizations of
neuroimaging outputs.
"""

from datetime import datetime
import io  # noqa: F401
import os
import re

from dipy.utils.logging import logger

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
            font-size: 14px;
            line-height: 1.55;
            color: #111827;
            background: #f5f6fa;
            margin: 0;
            padding: 0;
        }}

        /* ── Top bar ─────────────────────────────────────────── */
        .topbar {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 48px;
            background: #fff;
            border-bottom: 1px solid #e5e7eb;
            display: flex;
            align-items: center;
            padding: 0 20px;
            z-index: 200;
            gap: 24px;
        }}

        .topbar-brand {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 700;
            font-size: 15px;
            color: #111827;
            white-space: nowrap;
        }}

        .topbar-logo {{
            width: 28px;
            height: 28px;
            background: linear-gradient(135deg, #E87722 0%, #D94A38 100%);
            border-radius: 7px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 13px;
            font-weight: 800;
            flex-shrink: 0;
        }}

        .topbar-nav {{
            display: flex;
            gap: 4px;
            flex: 1;
        }}

        .topbar-nav a {{
            color: #6b7280;
            text-decoration: none;
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 13.5px;
            font-weight: 500;
            transition: background 0.15s, color 0.15s;
        }}

        .topbar-nav a:hover {{
            background: #f3f4f6;
            color: #111827;
        }}

        .topbar-nav a.active {{
            background: #E87722;
            color: white;
        }}

        .topbar-meta {{
            font-size: 12px;
            color: #9ca3af;
            white-space: nowrap;
        }}

        /* ── App shell ───────────────────────────────────────── */
        .app {{
            padding-top: 48px;
            display: flex;
            min-height: 100vh;
        }}

        /* ── Sidebar ─────────────────────────────────────────── */
        .sidebar {{
            width: 210px;
            min-width: 210px;
            background: #fff;
            border-right: 1px solid #e5e7eb;
            position: fixed;
            top: 48px;
            bottom: 0;
            overflow-y: auto;
            flex-shrink: 0;
            display: flex;
            flex-direction: column;
        }}

        .sidebar-nav {{
            padding: 12px 0 24px;
            flex: 1;
        }}

        .sidebar-section-label {{
            color: #9ca3af;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            padding: 14px 16px 4px;
            font-weight: 700;
        }}

        .sidebar-link {{
            display: block;
            color: #374151;
            text-decoration: none;
            padding: 6px 16px;
            font-size: 13.5px;
            transition: background 0.15s, color 0.15s;
            border-left: 2px solid transparent;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .sidebar-link:hover {{
            background: #f9fafb;
            color: #E87722;
            border-left-color: #E87722;
        }}

        .sidebar-sublink {{
            padding-left: 28px;
            font-size: 13px;
            color: #6b7280;
        }}

        .sidebar-sublink:hover {{
            color: #E87722;
        }}

        /* ── Main content ────────────────────────────────────── */
        .main-content {{
            flex: 1;
            min-width: 0;
            margin-left: 210px;
        }}

        .page-header {{
            padding: 28px 36px 22px;
            background: #fff;
            border-bottom: 1px solid #e5e7eb;
        }}

        .page-header h1 {{
            font-size: 22px;
            font-weight: 700;
            color: #111827;
            margin-bottom: 4px;
            padding: 0;
            border: none;
        }}

        .page-header p {{
            font-size: 13px;
            color: #6b7280;
        }}

        .info-banner {{
            margin: 0 36px 0;
            padding: 10px 16px;
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 8px;
            font-size: 13px;
            color: #92400e;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .content {{
            padding: 28px 36px 60px;
        }}

        /* ── Stat cards ──────────────────────────────────────── */
        .stat-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}

        .stat-card {{
            background: #fff;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 18px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }}

        .stat-card .stat-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.7px;
            color: #9ca3af;
            font-weight: 700;
            margin-bottom: 6px;
        }}

        .stat-card .stat-value {{
            font-size: 28px;
            font-weight: 700;
            color: #111827;
            line-height: 1;
        }}

        /* ── Section headings ────────────────────────────────── */
        section {{
            margin-bottom: 48px;
        }}

        h1 {{
            font-size: 18px;
            font-weight: 700;
            color: #111827;
            margin-bottom: 16px;
            padding-top: 28px;
            border: none;
        }}

        h2 {{
            font-size: 15px;
            font-weight: 600;
            color: #374151;
            margin-bottom: 10px;
            padding-top: 18px;
        }}

        h3 {{
            font-size: 13.5px;
            font-weight: 600;
            color: #4b5563;
            margin-bottom: 8px;
            padding-top: 12px;
        }}

        /* ── Info cards (config / summary) ───────────────────── */
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 14px;
            margin: 16px 0;
        }}

        .info-card {{
            background: #fff;
            padding: 16px 18px;
            border-radius: 10px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}

        .info-card strong {{
            display: block;
            color: #9ca3af;
            margin-bottom: 5px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.7px;
            font-weight: 700;
        }}

        .info-card p {{
            color: #111827;
            word-break: break-word;
            font-size: 13.5px;
        }}

        /* ── DAG ─────────────────────────────────────────────── */
        .dag-visualization {{
            background: #fff;
            border: 1px solid #e5e7eb;
            padding: 24px;
            border-radius: 10px;
            overflow-x: auto;
            margin: 16px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}

        .dag-visualization pre {{
            font-family: "SFMono-Regular", "Fira Code", "Consolas", monospace;
            font-size: 12.5px;
            line-height: 1.8;
            color: #374151;
        }}

        /* ── Table ───────────────────────────────────────────── */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            background: #fff;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}

        thead {{
            background: #f9fafb;
            border-bottom: 1px solid #e5e7eb;
        }}

        thead th {{
            color: #6b7280;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.7px;
            font-weight: 700;
        }}

        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #f3f4f6;
        }}

        tbody tr:last-child td {{
            border-bottom: none;
        }}

        tbody tr:hover {{
            background: #fafafa;
        }}

        /* ── Stage section ───────────────────────────────────── */
        .stage-section {{
            background: #fff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 22px 24px;
            margin: 20px 0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }}

        .stage-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
            padding-bottom: 14px;
            border-bottom: 1px solid #f3f4f6;
        }}

        .stage-title {{
            font-size: 15px;
            font-weight: 600;
            color: #111827;
        }}

        .stage-time {{
            background: #fff7ed;
            color: #9a3412;
            border: 1px solid #fed7aa;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }}

        /* ── Viewer / mosaic ─────────────────────────────────── */
        .viewer-container {{
            margin: 18px 0;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}

        .viewer-header {{
            background: #f9fafb;
            border-bottom: 1px solid #e5e7eb;
            color: #374151;
            padding: 10px 16px;
            font-weight: 600;
            font-size: 13px;
        }}

        canvas {{
            display: block;
            width: 100%;
            height: 500px;
            background: #000;
        }}

        /* ── Output items ────────────────────────────────────── */
        .output-list {{
            margin: 14px 0;
        }}

        .output-item {{
            background: #fff;
            padding: 13px 16px;
            margin: 8px 0;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }}

        .output-item strong {{
            color: #9ca3af;
            display: block;
            margin-bottom: 4px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            font-weight: 700;
        }}

        .output-item code {{
            background: #f3f4f6;
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 12.5px;
            color: #374151;
            font-family: "SFMono-Regular", "Fira Code", monospace;
            word-break: break-all;
        }}

        .download-link {{
            display: inline-block;
            margin-top: 10px;
            padding: 6px 14px;
            background: #E87722;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-size: 12.5px;
            font-weight: 600;
            transition: background 0.2s;
        }}

        .download-link:hover {{
            background: #c9621a;
        }}

        /* ── Footer ──────────────────────────────────────────── */
        footer {{
            background: #fff;
            border-top: 1px solid #e5e7eb;
            color: #9ca3af;
            padding: 20px 36px;
            text-align: center;
            font-size: 12px;
        }}

        /* ── Status boxes ────────────────────────────────────── */
        .error-box {{
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-radius: 8px;
            padding: 12px 16px;
            margin: 12px 0;
            color: #991b1b;
            font-size: 13.5px;
        }}

        .success-box {{
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 8px;
            padding: 12px 16px;
            margin: 12px 0;
            color: #166534;
            font-size: 13.5px;
        }}

        /* ── Methods / boilerplate ───────────────────────────── */
        .boilerplate {{
            background: #fff;
            border: 1px solid #e5e7eb;
            padding: 22px 24px;
            border-radius: 10px;
            margin: 14px 0;
            line-height: 1.75;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}

        .boilerplate p {{
            margin-bottom: 12px;
            color: #374151;
            font-size: 13.5px;
        }}

        .citation {{
            background: #f9fafb;
            border-left: 3px solid #E87722;
            border-radius: 0 6px 6px 0;
            padding: 13px 16px;
            margin: 10px 0;
            font-family: "SFMono-Regular", "Fira Code", monospace;
            font-size: 12px;
            color: #374151;
            line-height: 1.7;
        }}

        /* ── Badges ──────────────────────────────────────────── */
        .stage-badge {{
            display: inline-block;
            padding: 2px 9px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.2px;
            vertical-align: middle;
            margin-left: 8px;
        }}
        .badge-restart {{
            background: #eff6ff;
            color: #1d4ed8;
            border: 1px solid #bfdbfe;
        }}
        .badge-user-mask {{
            background: #f0fdf4;
            color: #166534;
            border: 1px solid #bbf7d0;
        }}

        /* ── Profile charts ──────────────────────────────────── */
        .profile-charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
            margin: 16px 0;
        }}
        .profile-chart-container {{
            background: #fff;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .profile-chart-container canvas {{
            height: 240px !important;
            background: #fff;
        }}

        @media print {{
            .topbar, .sidebar {{ display: none; }}
            .main-content {{ margin-left: 0; }}
            .viewer-container {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="topbar">
        <div class="topbar-brand">
            <div class="topbar-logo">D</div>
            DIPY Auto
        </div>
        <nav class="topbar-nav">
            <a href="#summary" class="active">Summary</a>
            <a href="#configuration">Configuration</a>
            <a href="#pipeline">Pipeline</a>
            <a href="#timing">Timing</a>
            <a href="#methods">Methods</a>
        </nav>
        <div class="topbar-meta">Generated {timestamp}</div>
    </div>

    <div class="app">
        <nav class="sidebar">
            <div class="sidebar-nav">
                <div class="sidebar-section-label">Report</div>
                <a href="#summary" class="sidebar-link">Summary</a>
                <a href="#configuration" class="sidebar-link">Configuration</a>
                <a href="#pipeline" class="sidebar-link">Pipeline</a>
                <a href="#timing" class="sidebar-link">Timing</a>
                <div class="sidebar-section-label">Stages</div>
                {sidebar_links}
                <div class="sidebar-section-label">Info</div>
                <a href="#methods" class="sidebar-link">Methods</a>
            </div>
        </nav>

        <div class="main-content">
            <div class="page-header">
                <h1>{title}</h1>
                <p>DIPY Auto Pipeline Report &mdash; {title_short}</p>
            </div>

            <div class="content">
                {content}
            </div>

            <footer>
                <p>Generated by DIPY Auto &mdash; Diffusion Imaging in Python</p>
            </footer>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
    <script>
        {inline_scripts}
    </script>
</body>
</html>
"""


def generate_summary_section(config, execution_info):
    """Generate summary section with key information."""
    general = config.get("General", {})
    io_config = config.get("io", {})

    num_stages = len(execution_info.get("stages", []))
    total_time = execution_info.get("total_time", 0)

    summary_html = """
    <section id="summary">
        <h1>Summary</h1>
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-label">Total Stages</div>
                <div class="stat-value">{num_stages}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Duration</div>
                <div class="stat-value">{total_time:.0f}
                  <span style="font-size:14px;font-weight:500;
                    color:#6b7280;margin-left:4px">s</span>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Status</div>
                <div class="stat-value"
                     style="font-size:20px;color:#16a34a">&#10003; Done</div>
            </div>
        </div>
        <div class="info-grid">
            <div class="info-card">
                <strong>Pipeline Name</strong>
                <p>{name}</p>
            </div>
            <div class="info-card">
                <strong>Description</strong>
                <p>{description}</p>
            </div>
            <div class="info-card">
                <strong>Version</strong>
                <p>{version}</p>
            </div>
            <div class="info-card">
                <strong>Author</strong>
                <p>{author}</p>
            </div>
            <div class="info-card">
                <strong>Output Directory</strong>
                <p><code>{out_dir}</code></p>
            </div>
        </div>
    </section>
    """.format(
        name=general.get("name", "Unknown"),
        description=general.get("description", "No description"),
        version=general.get("version", "N/A"),
        author=general.get("author", "N/A"),
        num_stages=num_stages,
        total_time=total_time,
        out_dir=io_config.get("out_dir", "."),
    )

    return summary_html


def generate_config_section(config, config_file_path):
    """Generate configuration section with links."""
    io_config = config.get("io", {})

    config_html = """
    <section id="configuration">
        <h1>Configuration</h1>
        <h2>Input Files</h2>
        <div class="output-list">
            <div class="output-item">
                <strong>DWI Image</strong>
                <code>{dwi}</code>
            </div>
            <div class="output-item">
                <strong>B-values</strong>
                <code>{bvals}</code>
            </div>
            <div class="output-item">
                <strong>B-vectors</strong>
                <code>{bvecs}</code>
            </div>
    """.format(
        dwi=io_config.get("dwi", "Not specified"),
        bvals=io_config.get("bvals", "Not specified"),
        bvecs=io_config.get("bvecs", "Not specified"),
    )

    if io_config.get("t1w"):
        config_html += """
            <div class="output-item">
                <strong>T1-weighted Image</strong>
                <code>{t1w}</code>
            </div>
        """.format(t1w=io_config["t1w"])

    config_html += """
        </div>
        <h2>Configuration File</h2>
        <div class="output-item">
            <strong>Config Path</strong>
            <code>{config_path}</code>
            <a href="{config_path}" class="download-link">View Config File</a>
        </div>
    </section>
    """.format(config_path=config_file_path)

    return config_html


def generate_pipeline_section(dag_visualization, execution_order):
    """Generate pipeline DAG and execution order section."""
    pipeline_html = """
    <section id="pipeline">
        <h1>Pipeline Execution Graph</h1>
        <p>The following diagram shows the dependency graph and execution order
        of pipeline stages:</p>
        <div class="dag-visualization">
            <pre>{dag}</pre>
        </div>
        <h2>Execution Order</h2>
        <p><strong>{order}</strong></p>
    </section>
    """.format(
        dag=dag_visualization.replace("<", "&lt;").replace(">", "&gt;"),
        order=" → ".join(execution_order),
    )

    return pipeline_html


def generate_timing_section(*, stages_info):
    """Generate timing table for all stages.

    Parameters
    ----------
    stages_info : list of dict
        Each dict must contain ``name``, and optionally ``cli``,
        ``duration``, ``success``, and ``skipped``. When ``skipped``
        is ``"restart"``, ``duration`` may be ``None``.

    Returns
    -------
    str
        HTML string for the timing section.
    """
    rows = ""
    for stage in stages_info:
        skipped = stage.get("skipped")
        duration = stage.get("duration")

        if skipped == "restart":
            duration_str = f"{duration:.2f}s" if duration is not None else "N/A"
            status_str = (
                '<span class="stage-badge badge-restart">Previous Run</span> ✓ Success'
            )
        elif skipped == "user_mask":
            duration_str = "0.00s (skipped)"
            status_str = (
                '<span class="stage-badge badge-user-mask">User Mask</span> ✓ Skipped'
            )
        else:
            duration_val = duration if duration is not None else 0.0
            duration_str = f"{duration_val:.2f}s"
            status_str = "✓ Success" if stage.get("success", True) else "✗ Failed"

        rows += """
        <tr>
            <td>{name}</td>
            <td>{cli}</td>
            <td>{duration}</td>
            <td>{status}</td>
        </tr>
        """.format(
            name=stage["name"],
            cli=stage.get("cli", "N/A"),
            duration=duration_str,
            status=status_str,
        )

    timing_html = """
    <section id="timing">
        <h1>Processing Time</h1>
        <table>
            <thead>
                <tr>
                    <th>Stage Name</th>
                    <th>CLI Command</th>
                    <th>Duration</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </section>
    """.format(rows=rows)

    return timing_html


def nifti_to_mosaic_svg(*, nifti_path, assets_dir, filename, n_cols=7, n_rows=3):
    """Render a 3-plane mosaic (axial / sagittal / coronal) of a NIfTI volume as SVG.

    Slices are drawn on a ``n_rows`` × ``n_cols`` grid.  The three rows correspond
    to axial, sagittal, and coronal planes respectively.  Slice positions are
    chosen via a density-projection bounding-box approach (nireports-style) so
    that the selected cuts are centred in brain tissue.  Each subplot carries
    orientation labels (L/R or A/P) at the top corners and a world-space
    coordinate label (e.g. ``z=-16``) at the bottom-left.

    Parameters
    ----------
    nifti_path : str
        Path to the NIfTI file.
    assets_dir : str
        Directory where the SVG file is saved.
    filename : str
        Output filename (e.g. ``stage_1_out_fa.svg``).
    n_cols : int, optional
        Number of columns (cuts per anatomical plane).
    n_rows : int, optional
        Number of rows.  Must be 3 (one per anatomical plane); provided for
        API compatibility.

    Returns
    -------
    svg_path : pathlib.Path or None
        Path to the saved SVG, or ``None`` if generation failed.
    """
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np

    try:
        img = nib.load(str(nifti_path))
        img = nib.as_closest_canonical(img)
        affine = img.affine
        data = img.get_fdata(dtype=np.float32)
    except Exception:
        return None

    if data.ndim == 4:
        data = data[..., data.shape[3] // 2]
    if data.ndim != 3:
        return None

    nonzero = data[data > 0]
    if nonzero.size == 0:
        return None
    vmin, vmax = np.percentile(nonzero, [2, 98])
    if vmax - vmin < 1e-6:  # binary or near-constant
        vmin = 0.0
        vmax = float(nonzero.max()) if nonzero.max() > 0 else 1.0

    # --- density-projection bounding box (nireports cuts_from_bbox style) ---
    low_th = float(np.percentile(nonzero, 5))
    if low_th >= float(nonzero.max()):  # binary: percentile equals max
        low_th = 0.0
    mask = data > low_th
    nx, ny, nz = mask.shape

    density = [
        mask.sum(axis=(1, 2)),  # axis 0 — sagittal, shape (nx,)
        mask.sum(axis=(0, 2)),  # axis 1 — coronal,  shape (ny,)
        mask.sum(axis=(0, 1)),  # axis 2 — axial,    shape (nz,)
    ]
    thresholds = [
        int(np.ceil((ny * nz) * 0.2)),  # sagittal: 20 %
        int(np.ceil((nx * nz) * 0.1)),  # coronal:  10 %
        int(np.ceil((nx * ny) * 0.3)),  # axial:    30 %
    ]

    indices = {}
    for ax_idx, (dens, th) in enumerate(zip(density, thresholds)):
        cands = np.argwhere(dens > th).ravel()
        if cands.size < 2:
            cands = np.argwhere(dens > 0).ravel()
        s_min = int(cands[0]) if cands.size else 0
        s_max = int(cands[-1]) if cands.size else data.shape[ax_idx] - 1
        indices[ax_idx] = np.linspace(s_min, s_max, num=n_cols + 2)[1:-1].astype(int)

    # row order: axial (axis 2), sagittal (axis 0), coronal (axis 1)
    ROW_AXES = [2, 0, 1]
    LEFT_LABEL = {2: "R", 0: "A", 1: "R"}
    RIGHT_LABEL = {2: "L", 0: "P", 1: "L"}
    AXIS_LETTER = {0: "x", 1: "y", 2: "z"}

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2.2, n_rows * 2.6),
    )
    fig.patch.set_facecolor("black")

    for row, ax_idx in enumerate(ROW_AXES):
        letter = AXIS_LETTER[ax_idx]
        for col, idx in enumerate(indices[ax_idx]):
            ax_obj = axes[row, col]
            slc = np.take(data, idx, axis=ax_idx)
            ax_obj.imshow(
                slc.T,
                cmap="gray",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
                interpolation="nearest",
            )
            ax_obj.set_axis_off()

            # world-space coordinate label
            v = np.zeros(4)
            v[ax_idx] = idx
            v[3] = 1
            wcoord = round(float((affine @ v)[ax_idx]))
            coord_label = f"{letter}={wcoord}"

            # orientation labels (top corners)
            ax_obj.text(
                0.02,
                0.98,
                LEFT_LABEL[ax_idx],
                color="white",
                fontsize=7,
                ha="left",
                va="top",
                transform=ax_obj.transAxes,
            )
            ax_obj.text(
                0.98,
                0.98,
                RIGHT_LABEL[ax_idx],
                color="white",
                fontsize=7,
                ha="right",
                va="top",
                transform=ax_obj.transAxes,
            )
            # coordinate label (bottom-left)
            ax_obj.text(
                0.02,
                0.02,
                coord_label,
                color="white",
                fontsize=7,
                ha="left",
                va="bottom",
                transform=ax_obj.transAxes,
            )

    fig.tight_layout(pad=0.1)
    svg_path = Path(assets_dir) / filename
    fig.savefig(
        str(svg_path),
        format="svg",
        bbox_inches="tight",
        facecolor="black",
        dpi=150,
    )
    plt.close(fig)
    return svg_path


def generate_mosaic_viewer(
    *, canvas_id, nifti_path, viewer_title, assets_dir, html_dir
):
    """Generate an ``<img>`` tag pointing to a saved SVG mosaic.

    Parameters
    ----------
    canvas_id : str
        Unique ID used to build the SVG filename.
    nifti_path : str
        Path to the NIfTI file.
    viewer_title : str
        Caption shown above the mosaic.
    assets_dir : str
        Directory where the SVG is saved.
    html_dir : str
        Directory of the HTML report (for computing relative paths).

    Returns
    -------
    html : str
        HTML fragment with the mosaic image.
    """
    filename = re.sub(r"[^A-Za-z0-9_-]", "_", canvas_id) + ".svg"
    svg_path = nifti_to_mosaic_svg(
        nifti_path=nifti_path, assets_dir=assets_dir, filename=filename
    )
    if svg_path is None:
        return ""

    rel_path = os.path.relpath(str(svg_path), html_dir)
    return (
        '<div class="viewer-container">'
        f'<div class="viewer-header">{viewer_title}</div>'
        f'<img src="{rel_path}" '
        'style="width:100%;background:#000;display:block;">'
        "</div>"
    )


def is_viewable_format(file_path):
    """Check if file is a NIfTI-family volume renderable as a slice mosaic."""
    if not file_path or not os.path.exists(str(file_path)):
        return False
    p = str(file_path).lower()
    if p.endswith(".nii.gz"):
        return True
    return os.path.splitext(p)[1] in {".nii", ".nrrd", ".mgh", ".mgz"}


def generate_methods_section(stages_info, pipeline_name):
    """Generate methods section with workflow citations.

    Parameters
    ----------
    stages_info : list
        List of stage information dictionaries.
    pipeline_name : str
        Name of the pipeline.
    """
    from dipy.workflows.cli import cli_flows

    methods_html = """
    <section id="methods">
        <h1>Methods</h1>
        <div class="boilerplate">
            <p>Results included in this report were generated using DIPY Auto
            (Diffusion Imaging in Python), a comprehensive pipeline for diffusion
            MRI data processing and analysis.</p>

            <h2>Pipeline: {pipeline}</h2>
            <p>The following processing steps were applied:</p>
    """.format(pipeline=pipeline_name)

    unique_clis = set()
    stage_descriptions = []

    for stage in stages_info:
        cli_name = stage.get("cli")
        stage_name = stage["name"]

        if cli_name and cli_name in cli_flows:
            unique_clis.add(cli_name)
            module_name, class_name = cli_flows[cli_name]

            try:
                import importlib

                module = importlib.import_module(module_name)
                workflow_class = getattr(module, class_name)
                doc = workflow_class.__doc__ or ""
                first_line = doc.strip().split("\n")[0] if doc else cli_name

                stage_descriptions.append(
                    f"<li><strong>{stage_name}</strong>: "
                    f"{first_line} ({cli_name})</li>"
                )
            except Exception:
                stage_descriptions.append(
                    f"<li><strong>{stage_name}</strong>: {cli_name}</li>"
                )

    if stage_descriptions:
        methods_html += "<ol>" + "\n".join(stage_descriptions) + "</ol>"

    methods_html += """
            <h2>References</h2>
            <p>Please cite the following references when using DIPY Auto:</p>
            <div class="citation">
    """

    dipy_citation = """Garyfallidis E, Brett M, Amirbekian B, Rokem A, van der Walt S,
Descoteaux M, Nimmo-Smith I, and Dipy Contributors (2014).
DIPY, a library for the analysis of diffusion MRI data.
Frontiers in Neuroinformatics, vol.8, no.8.
https://doi.org/10.3389/fninf.2014.00008"""

    methods_html += dipy_citation
    methods_html += """
            </div>
        </div>
    </section>
    """

    return methods_html


def generate_buan_profiles_charts(*, out_dir):
    """Generate Chart.js line plots for BUAN along-tract profiles.

    Parameters
    ----------
    out_dir : str
        Directory produced by the ``buan_profiles`` stage.

    Returns
    -------
    html : str
        HTML fragment containing canvas elements.
    js : str
        JavaScript snippet initialising Chart.js charts.
    """
    from pathlib import Path

    import numpy as np

    npy_files = sorted(Path(out_dir).glob("*_profile.npy"))
    if not npy_files:
        return "", ""

    bundle_profiles = {}
    for f in npy_files:
        core = f.stem[:-8]  # strip "_profile"
        sep = core.rfind("_")
        bundle = core[:sep]
        metric = core[sep + 1 :]
        raw = np.load(f).astype(float)
        mean, std = np.nanmean(raw), np.nanstd(raw)
        z = (raw - mean) / std if std > 0 else raw - mean
        max_abs = np.nanmax(np.abs(z))
        normalized = z / max_abs if max_abs > 0 else z
        bundle_profiles.setdefault(bundle, {})[metric] = normalized

    COLORS = [
        "#E87722",
        "#3498db",
        "#2ecc71",
        "#9b59b6",
        "#e74c3c",
        "#1abc9c",
        "#f39c12",
        "#e67e22",
    ]

    # Collect ordered unique metrics across all bundles
    all_metrics = sorted({m for metrics in bundle_profiles.values() for m in metrics})
    metric_color = {m: COLORS[i % len(COLORS)] for i, m in enumerate(all_metrics)}

    charts_html_parts = []
    js_parts = []

    for bundle, metrics in sorted(bundle_profiles.items()):
        canvas_id = "buan_" + re.sub(r"[^A-Za-z0-9_]", "_", bundle)
        n_points = len(next(iter(metrics.values())))
        labels = list(range(n_points))

        datasets = []
        for metric in all_metrics:
            if metric not in metrics:
                continue
            profile = metrics[metric]
            color = metric_color[metric]
            data_vals = [
                "null" if np.isnan(v) else str(round(float(v), 6)) for v in profile
            ]
            datasets.append(
                f'{{"label":"{metric}",'
                f'"data":[{",".join(data_vals)}],'
                f'"borderColor":"{color}",'
                f'"backgroundColor":"{color}22",'
                f'"fill":false,"tension":0.3,"pointRadius":0,"borderWidth":2}}'
            )

        charts_html_parts.append(
            f'<div class="profile-chart-container">'
            f'<canvas id="{canvas_id}"></canvas>'
            f"</div>"
        )

        js_parts.append(f"""
(function(){{
  var ctx=document.getElementById('{canvas_id}').getContext('2d');
  var chart=new Chart(ctx,{{
    type:'line',
    data:{{
      labels:{labels},
      datasets:[{",".join(datasets)}]
    }},
    options:{{
      responsive:true,
      maintainAspectRatio:false,
      plugins:{{
        title:{{display:true,text:'{bundle}',font:{{size:14,weight:'bold'}}}},
        legend:{{display:true,position:'top'}}
      }},
      scales:{{
        x:{{title:{{display:true,text:'% Distance Along Bundle'}}}},
        y:{{title:{{display:true,text:'Normalized value (z-score)'}},min:-1,max:1}}
      }}
    }}
  }});
  if(typeof buanCharts==='undefined'){{window.buanCharts=[];}}
  window.buanCharts.push(chart);
}})();""")

    # Build global-toggle checkboxes, one per metric
    checkbox_items = []
    for metric in all_metrics:
        color = metric_color[metric]
        cb_id = f"buan_toggle_{re.sub(r'[^A-Za-z0-9_]', '_', metric)}"
        checkbox_items.append(
            f'<label for="{cb_id}" style="display:inline-flex;align-items:center;'
            f'gap:5px;margin-right:14px;cursor:pointer;font-size:0.88em;">'
            f'<input type="checkbox" id="{cb_id}" checked '
            f'class="buan-metric-toggle" data-metric="{metric}" '
            f'style="accent-color:{color};width:15px;height:15px;">'
            f'<span style="color:{color};font-weight:600;">{metric}</span>'
            f"</label>"
        )

    toggle_bar = (
        '<div style="margin:12px 0 4px;padding:10px 14px;background:#f8f9fa;'
        'border-radius:6px;border:1px solid #e0e0e0;">'
        '<span style="font-size:0.85em;color:#555;margin-right:12px;'
        'font-weight:600;">Show/hide all charts:</span>'
        + "".join(checkbox_items)
        + "</div>"
    )

    toggle_js = """
(function(){
  window.buanCharts = window.buanCharts || [];
  document.querySelectorAll('.buan-metric-toggle').forEach(function(cb){
    cb.addEventListener('change', function(){
      var metric = this.dataset.metric;
      var visible = this.checked;
      window.buanCharts.forEach(function(chart){
        chart.data.datasets.forEach(function(ds, i){
          if(ds.label === metric){ chart.setDatasetVisibility(i, visible); }
        });
        chart.update();
      });
    });
  });
})();"""

    html = (
        "<h3>Along-tract Profiles</h3>"
        + toggle_bar
        + '<div class="profile-charts-grid">'
        + "".join(charts_html_parts)
        + "</div>"
    )
    return html, "\n".join(js_parts) + toggle_js


def generate_results_section(*, stages_info, html_dir, assets_dir):
    """Generate results section with slice mosaics for each stage.

    Parameters
    ----------
    stages_info : list of dict
        List of stage information dictionaries. Each dict must contain
        ``name`` and optionally ``cli``, ``duration``, ``outputs``,
        and ``skipped``.
    html_dir : str
        Directory where the HTML file is saved (for computing relative
        paths).
    assets_dir : str
        Directory where SVG mosaic files are saved.

    Returns
    -------
    tuple of (str, str)
        HTML string for the results section and a string of Chart.js
        JavaScript snippets to embed (BUAN charts only).
    """
    results_html = '<section id="results"><h1>Stage Results</h1>'
    buan_scripts = []

    for idx, stage in enumerate(stages_info):
        stage_name = stage["name"]
        cli = stage.get("cli", "Unknown")
        duration = stage.get("duration")
        outputs = stage.get("outputs", {})
        skipped = stage.get("skipped")

        if skipped == "restart":
            duration_str = f"{duration:.2f}s" if duration is not None else "N/A"
            badge_html = '<span class="stage-badge badge-restart">Previous Run</span>'
        elif skipped == "user_mask":
            duration_str = "Skipped"
            badge_html = '<span class="stage-badge badge-user-mask">User Mask</span>'
        else:
            duration_val = duration if duration is not None else 0.0
            duration_str = f"{duration_val:.2f}s"
            badge_html = ""

        stage_anchor = "stage_" + re.sub(r"[^A-Za-z0-9_-]", "_", stage_name)
        results_html += """
        <div class="stage-section" id="{anchor}">
            <div class="stage-header">
                <div class="stage-title">Stage {num}: {name}{badge}</div>
                <div class="stage-time">{duration}</div>
            </div>
            <p><strong>CLI Command:</strong> <code>{cli}</code></p>
        """.format(
            anchor=stage_anchor,
            num=idx + 1,
            name=stage_name,
            badge=badge_html,
            duration=duration_str,
            cli=cli,
        )

        if outputs:
            results_html += '<h3>Output Files</h3><div class="output-list">'

            for output_param, output_path in outputs.items():
                results_html += """
                <div class="output-item">
                    <strong>{param}</strong>
                    <code>{path}</code>
                    <a href="{path}" class="download-link">Download</a>
                </div>
                """.format(param=output_param, path=output_path)

                if is_viewable_format(output_path):
                    viewer_id = f"nv_stage_{idx}_{output_param}"
                    viewer_html = generate_mosaic_viewer(
                        canvas_id=viewer_id,
                        nifti_path=output_path,
                        viewer_title=f"{output_param}: {os.path.basename(output_path)}",
                        assets_dir=assets_dir,
                        html_dir=html_dir,
                    )
                    results_html += viewer_html

            results_html += "</div>"
        else:
            results_html += "<p><em>No outputs recorded for this stage.</em></p>"

        if stage_name == "buan_profiles":
            buan_out = outputs.get("out_dir") or stage.get("params", {}).get("out_dir")
            if buan_out and os.path.isdir(buan_out):
                charts_html, charts_js = generate_buan_profiles_charts(out_dir=buan_out)
                results_html += charts_html
                buan_scripts.append(charts_js)

        results_html += "</div>"

    results_html += "</section>"

    return results_html, "\n".join(buan_scripts)


def generate_sidebar(*, stages_info):
    """Generate sidebar navigation links for all pipeline stages.

    Parameters
    ----------
    stages_info : list of dict
        List of stage information dictionaries with at least a ``name`` key.

    Returns
    -------
    str
        HTML fragment with ``<a>`` links for the sidebar.
    """
    links = []
    for stage in stages_info:
        name = stage["name"]
        anchor = "stage_" + re.sub(r"[^A-Za-z0-9_-]", "_", name)
        links.append(
            f'<a href="#{anchor}" class="sidebar-link sidebar-sublink">{name}</a>'
        )
    return "\n".join(links)


def generate_html_report(config, config_file_path, execution_info, output_path):
    """Generate complete HTML report.

    Parameters
    ----------
    config : dict
        Pipeline configuration dictionary.
    config_file_path : str
        Path to the configuration file.
    execution_info : dict
        Execution information including:
        - stages: list of stage info dicts
        - total_time: total execution time
        - dag_visualization: DAG text visualization
        - execution_order: list of stage names in execution order
    output_path : str
        Path where HTML report should be saved.
    """
    logger.info(
        "Generating HTML report — rendering slice mosaics for each NIfTI "
        "output, this may take a few minutes..."
    )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pipeline_name = config.get("General", {}).get("name", "Pipeline")
    title = f"DIPY Auto Pipeline Report - {pipeline_name}"

    html_dir = os.path.dirname(os.path.abspath(output_path))
    stages = execution_info.get("stages", [])

    assets_dir = os.path.join(html_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    summary = generate_summary_section(config, execution_info)
    configuration = generate_config_section(config, config_file_path)
    pipeline = generate_pipeline_section(
        execution_info.get("dag_visualization", ""),
        execution_info.get("execution_order", []),
    )
    timing = generate_timing_section(stages_info=stages)
    results, inline_scripts = generate_results_section(
        stages_info=stages,
        html_dir=html_dir,
        assets_dir=assets_dir,
    )
    methods = generate_methods_section(stages, pipeline_name)
    sidebar_links = generate_sidebar(stages_info=stages)

    content = summary + configuration + pipeline + timing + results + methods

    html = HTML_TEMPLATE.format(
        title=title,
        title_short=pipeline_name,
        timestamp=timestamp,
        content=content,
        inline_scripts=inline_scripts,
        sidebar_links=sidebar_links,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
