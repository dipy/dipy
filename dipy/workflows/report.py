"""HTML reports for dipy_auto pipeline execution.

Every stage gets its own self-contained QC page (``reports/NN_<stage>.html``)
written as soon as the stage finishes, plus a JSON record next to it. An
overview page (``reports/index.html``) is refreshed after each stage, and the
final report combines all stage records. Figures are embedded as data URIs so
every page opens with a double-click, without a web server.
"""

from datetime import datetime
import html
import importlib
import json
import os
import re

from dipy.utils.logging import logger
from dipy.workflows.cli import cli_flows
from dipy.workflows.qc import STATUS_LEVELS

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
        .badge-optional {{
            background: #f8fafc;
            color: #475569;
            border: 1px solid #cbd5e1;
        }}
        /* ── QC ──────────────────────────────────────────────── */
        .badge-pass {{
            background: #f0fdf4;
            color: #166534;
            border: 1px solid #bbf7d0;
        }}
        .badge-info, .badge-pending {{
            background: #f3f4f6;
            color: #4b5563;
            border: 1px solid #e5e7eb;
        }}
        .badge-warn {{
            background: #fffbeb;
            color: #92400e;
            border: 1px solid #fde68a;
        }}
        .badge-fail, .badge-error {{
            background: #fef2f2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }}
        .metric-value {{
            font-family: "SFMono-Regular", "Fira Code", monospace;
            font-size: 12.5px;
            white-space: pre;
        }}
        .error-box {{
            white-space: pre-wrap;
            font-family: "SFMono-Regular", "Fira Code", monospace;
            font-size: 12px;
        }}
        .figure-img {{
            width: 100%;
            display: block;
            background: #000;
        }}
        .sidebar-link.current {{
            color: #E87722;
            border-left-color: #E87722;
            font-weight: 600;
        }}

        /* ── Open dialog ─────────────────────────────────────── */
        .qc-open {{
            display: inline-block;
            margin: 10px 8px 0 0;
            padding: 6px 14px;
            background: #E87722;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 12.5px;
            font-weight: 600;
            font-family: inherit;
            cursor: pointer;
        }}
        .qc-open:hover {{
            background: #c9621a;
        }}
        .qc-dialog {{
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(17,24,39,0.45);
            z-index: 300;
            align-items: center;
            justify-content: center;
        }}
        .qc-dialog-box {{
            background: #fff;
            border-radius: 12px;
            padding: 22px 24px;
            max-width: 560px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .qc-dialog-box code {{
            display: block;
            background: #f3f4f6;
            padding: 10px 12px;
            border-radius: 6px;
            margin: 12px 0;
            font-size: 12.5px;
            word-break: break-all;
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
            {nav}
        </nav>
        <div class="topbar-meta">Generated {timestamp}</div>
    </div>

    <div class="app">
        <nav class="sidebar">
            <div class="sidebar-nav">
                {sidebar}
            </div>
        </nav>

        <div class="main-content">
            <div class="page-header">
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>

            <div class="content">
                {content}
            </div>

            <footer>
                <p>Generated by DIPY Auto &mdash; Diffusion Imaging in Python</p>
            </footer>
        </div>
    </div>

    <div class="qc-dialog" id="qc-dialog">
        <div class="qc-dialog-box">
            <h2 id="qc-dialog-title">Skyline was not found</h2>
            <p id="qc-dialog-msg"></p>
            <code id="qc-dialog-cmd"></code>
            <button class="qc-open" onclick="qcCopy()">Copy command</button>
            <button class="qc-open" onclick="qcClose()">Close</button>
        </div>
    </div>

    <script>
        function qcDialog(message, path) {{
            document.getElementById('qc-dialog-msg').textContent = message;
            document.getElementById('qc-dialog-cmd').textContent =
                'dipy_skyline ' + path;
            document.getElementById('qc-dialog').style.display = 'flex';
        }}
        function qcClose() {{
            document.getElementById('qc-dialog').style.display = 'none';
        }}
        function qcCopy() {{
            var text = document.getElementById('qc-dialog-cmd').textContent;
            try {{
                navigator.clipboard.writeText(text);
            }} catch (e) {{
                window.prompt('Copy the command:', text);
            }}
        }}
        function qcOpen(path) {{
            fetch('/qc-open?path=' + encodeURIComponent(path))
                .then(function (r) {{
                    if (!r.ok) {{
                        return r.text().then(function (t) {{ qcDialog(t, path); }});
                    }}
                }})
                .catch(function () {{
                    qcDialog(
                        'Skyline can only be launched when this report is served. ' +
                        'Run "dipy_auto <report.html>" and use the button from there, ' +
                        'or run the command below yourself.',
                        path
                    );
                }});
        }}
    </script>
</body>
</html>
"""

# Formats the Skyline viewer can load; anything else opens in the browser.
SKYLINE_FORMATS = (
    ".nii",
    ".nii.gz",
    ".trx",
    ".trk",
    ".tck",
    ".pam5",
    ".vtk",
    ".vtp",
    ".fib",
    ".dpy",
    ".nrrd",
    ".mgh",
    ".mgz",
)


# =============================================================================
# Shared building blocks
# =============================================================================


def stage_basename(*, index, stage_name):
    """File name stem of a stage page, e.g. ``03_brain_mask``.

    Parameters
    ----------
    index : int
        1-based position of the stage in the pipeline.
    stage_name : str
        Stage name.

    Returns
    -------
    str
        Zero-padded stem so files sort in pipeline order.
    """
    return f"{index:02d}_{re.sub(r'[^A-Za-z0-9_-]', '_', stage_name)}"


def _badge(status):
    return f'<span class="stage-badge badge-{status.lower()}">{status}</span>'


def _run_badge(record):
    skipped = record.get("skipped")
    if skipped == "restart":
        return '<span class="stage-badge badge-restart">Previous Run</span>'
    if skipped == "user_mask":
        return '<span class="stage-badge badge-user-mask">User Mask</span>'
    if skipped == "optional":
        reason = html.escape(record.get("reason", ""), quote=True)
        return (
            f'<span class="stage-badge badge-optional" title="{reason}">Skipped</span>'
        )
    return ""


def _duration(record):
    duration = record.get("duration")
    return f"{duration:.2f}s" if duration is not None else "N/A"


def _links(links, *, css="sidebar-link", current=None):
    return "\n".join(
        f'<a href="{href}" class="{css}{" current" if href == current else ""}">'
        f"{html.escape(label)}</a>"
        for href, label in links
    )


def _page(*, title, subtitle, nav, sidebar, content):
    return HTML_TEMPLATE.format(
        title=html.escape(title),
        subtitle=html.escape(subtitle),
        nav=_links(nav, css=""),
        sidebar=sidebar,
        content=content,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def _format_value(*, value, unit):
    text = f"{value:.4g}" if isinstance(value, float) else str(value)
    return html.escape(f"{text} {unit}".strip())


def _output_items(*, outputs, page_dir):
    """Output list of a stage, each file with a button that opens it.

    A file Skyline can display gets a button that launches it there, which
    needs the report to be served; every other file is opened by the browser
    through a plain link.

    Parameters
    ----------
    outputs : dict
        Mapping of output parameter name to file path.
    page_dir : str
        Directory of the page being written, for relative links.

    Returns
    -------
    str
        HTML fragment.
    """
    items = ""
    for key, path in outputs.items():
        path = str(path)
        if key == "out_dir" or not os.path.exists(path):
            button = ""
        elif path.endswith(SKYLINE_FORMATS):
            button = (
                f'<button class="qc-open" '
                f"onclick=\"qcOpen('{html.escape(path, quote=True)}')\">"
                f"Open in Skyline</button>"
            )
        else:
            href = os.path.relpath(path, page_dir)
            button = f'<a class="qc-open" href="{href}" target="_blank">Open</a>'
        items += (
            f'<div class="output-item"><strong>{html.escape(key)}</strong>'
            f"<code>{html.escape(path)}</code>{button}</div>"
        )
    return items


def render_stage_section(*, record, number, anchor, page_dir):
    """HTML card of one stage: QC metrics, figures, errors and outputs.

    Parameters
    ----------
    record : dict
        Stage record: stage info (``name``, ``cli``, ``duration``,
        ``outputs``, ``skipped``) plus the QC result under ``qc``.
    number : int
        1-based stage position shown in the title.
    anchor : str
        HTML id of the card.
    page_dir : str
        Directory of the page being written, for relative links.

    Returns
    -------
    str
        HTML fragment.
    """
    qc = record["qc"]
    rows = "".join(
        f"<tr><td>{html.escape(m['name'])}</td>"
        f'<td class="metric-value">{_format_value(value=m["value"], unit=m["unit"])}</td>'
        f"<td>{html.escape(m['rule'])}</td><td>{_badge(m['status'])}</td></tr>"
        for m in qc["metrics"]
    )
    metrics = (
        "<h3>QC Metrics</h3><table><thead><tr><th>Metric</th><th>Value</th>"
        f"<th>Rule</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table>"
        if rows
        else ""
    )
    figures = "".join(
        f'<div class="viewer-container"><div class="viewer-header">'
        f"{html.escape(f['title'])}</div>"
        f'<img class="figure-img" src="{f["data_uri"]}" alt=""></div>'
        for f in qc["figures"]
    )
    error = (
        f'<pre class="error-box">{html.escape(qc["error"])}</pre>'
        if qc["error"]
        else ""
    )
    outputs = _output_items(outputs=record.get("outputs", {}), page_dir=page_dir)
    if outputs:
        outputs = f'<h3>Output Files</h3><div class="output-list">{outputs}</div>'
    return f"""
    <div class="stage-section" id="{anchor}">
        <div class="stage-header">
            <div class="stage-title">Stage {number}: {html.escape(record["name"])}
                {_badge(qc["status"])}{_run_badge(record)}</div>
            <div class="stage-time">{_duration(record)}</div>
        </div>
        <p><strong>CLI Command:</strong> <code>{record.get("cli", "N/A")}</code></p>
        {error}{metrics}{figures}{outputs}
    </div>
    """


def _overview_section(*, stage_names, records, href_prefix):
    rows = ""
    for index, (name, record) in enumerate(zip(stage_names, records), start=1):
        if record is None:
            rows += (
                f"<tr><td>{index}</td><td>{html.escape(name)}</td><td></td><td></td>"
                f"<td>{_badge('PENDING')}</td><td></td></tr>"
            )
            continue
        href = f"{href_prefix}{stage_basename(index=index, stage_name=name)}.html"
        rows += (
            f"<tr><td>{index}</td><td>{html.escape(name)}{_run_badge(record)}</td>"
            f"<td><code>{record.get('cli', '')}</code></td>"
            f"<td>{_duration(record)}</td><td>{_badge(record['qc']['status'])}</td>"
            f'<td><a href="{href}">open</a></td></tr>'
        )
    return f"""
    <section id="overview">
        <h1>Stages Overview</h1>
        <table>
            <thead><tr><th>#</th><th>Stage</th><th>CLI Command</th>
            <th>Duration</th><th>QC</th><th>Report</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </section>
    """


def load_stage_record(*, reports_dir, index, stage_name):
    """Load the JSON record of one stage.

    Parameters
    ----------
    reports_dir : str
        Directory holding the stage pages.
    index : int
        1-based position of the stage.
    stage_name : str
        Stage name.

    Returns
    -------
    dict or None
        Stage record, ``None`` if the stage has no record yet.
    """
    path = os.path.join(
        reports_dir, stage_basename(index=index, stage_name=stage_name) + ".json"
    )
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_stage_records(*, reports_dir, stage_names):
    """Load the JSON record of each stage.

    Parameters
    ----------
    reports_dir : str
        Directory holding the stage pages.
    stage_names : list of str
        Stage names in pipeline order.

    Returns
    -------
    list
        One record dict per stage, ``None`` for stages without a record.
    """
    return [
        load_stage_record(reports_dir=reports_dir, index=index, stage_name=name)
        for index, name in enumerate(stage_names, start=1)
    ]


def _write(*, path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# =============================================================================
# Stage pages
# =============================================================================


def write_stage_report(*, reports_dir, stage_names, index, record, pipeline_name):
    """Write the JSON record and the standalone HTML page of one stage.

    Parameters
    ----------
    reports_dir : str
        Output directory of the stage pages.
    stage_names : list of str
        All stage names in pipeline order, for navigation.
    index : int
        1-based position of this stage.
    record : dict
        Stage record, see ``render_stage_section``.
    pipeline_name : str
        Pipeline name shown in the page header.

    Returns
    -------
    str
        Path of the HTML page.
    """
    base = stage_basename(index=index, stage_name=record["name"])
    _write(
        path=os.path.join(reports_dir, base + ".json"),
        text=json.dumps(record, default=str),
    )

    pages = [
        (f"{stage_basename(index=i, stage_name=n)}.html", f"{i}. {n}")
        for i, n in enumerate(stage_names, start=1)
    ]
    nav = [("index.html", "Overview")]
    if index > 1:
        nav.insert(0, (pages[index - 2][0], "← Previous"))
    if index < len(pages):
        nav.append((pages[index][0], "Next →"))
    sidebar = (
        '<div class="sidebar-section-label">Report</div>'
        + _links([("index.html", "Overview")])
        + '<div class="sidebar-section-label">Stages</div>'
        + _links(pages, css="sidebar-link sidebar-sublink", current=base + ".html")
    )
    page = _page(
        title=f"Stage {index}: {record['name']}",
        subtitle=f"DIPY Auto QC — {pipeline_name}",
        nav=nav,
        sidebar=sidebar,
        content=render_stage_section(
            record=record, number=index, anchor="stage", page_dir=reports_dir
        ),
    )
    return _write(path=os.path.join(reports_dir, base + ".html"), text=page)


def write_index(*, reports_dir, stage_names, pipeline_name):
    """Write ``index.html``, the live overview of all stage pages.

    Parameters
    ----------
    reports_dir : str
        Output directory of the stage pages.
    stage_names : list of str
        All stage names in pipeline order.
    pipeline_name : str
        Pipeline name shown in the page header.

    Returns
    -------
    str
        Path of the index page.
    """
    records = load_stage_records(reports_dir=reports_dir, stage_names=stage_names)
    pages = [
        (f"{stage_basename(index=i, stage_name=n)}.html", f"{i}. {n}")
        for i, (n, r) in enumerate(zip(stage_names, records), start=1)
        if r is not None
    ]
    sidebar = '<div class="sidebar-section-label">Stages</div>' + _links(
        pages, css="sidebar-link sidebar-sublink"
    )
    page = _page(
        title=f"QC Overview - {pipeline_name}",
        subtitle="DIPY Auto QC — updated after every stage",
        nav=[("index.html", "Overview")],
        sidebar=sidebar,
        content=_overview_section(
            stage_names=stage_names, records=records, href_prefix=""
        ),
    )
    return _write(path=os.path.join(reports_dir, "index.html"), text=page)


# =============================================================================
# Final report
# =============================================================================


def generate_summary_section(*, config, execution_info, records):
    """Generate summary section with key information and overall QC status."""
    general = config.get("General", {})
    statuses = [r["qc"]["status"] for r in records if r is not None]
    worst = max(statuses, key=STATUS_LEVELS.index, default="INFO")

    return """
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
                <div class="stat-label">Worst QC Status</div>
                <div class="stat-value" style="font-size:20px">{worst}</div>
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
        num_stages=len(execution_info.get("stages", [])),
        total_time=execution_info.get("total_time", 0),
        worst=_badge(worst),
        out_dir=config.get("io", {}).get("out_dir", "."),
    )


def generate_config_section(*, config, config_file_path):
    """Generate configuration section with links."""
    io_config = config.get("io", {})
    inputs = [
        ("DWI Image", io_config.get("dwi", "Not specified")),
        ("B-values", io_config.get("bvals", "Not specified")),
        ("B-vectors", io_config.get("bvecs", "Not specified")),
    ]
    if io_config.get("t1w"):
        inputs.append(("T1-weighted Image", io_config["t1w"]))
    items = "".join(
        f'<div class="output-item"><strong>{label}</strong><code>{value}</code></div>'
        for label, value in inputs
    )
    return f"""
    <section id="configuration">
        <h1>Configuration</h1>
        <h2>Input Files</h2>
        <div class="output-list">{items}</div>
        <h2>Configuration File</h2>
        <div class="output-item">
            <strong>Config Path</strong>
            <code>{config_file_path}</code>
            <a href="{config_file_path}" class="download-link">View Config File</a>
        </div>
    </section>
    """


def generate_pipeline_section(*, dag_visualization, execution_order):
    """Generate pipeline DAG and execution order section."""
    return f"""
    <section id="pipeline">
        <h1>Pipeline Execution Graph</h1>
        <p>The following diagram shows the dependency graph and execution order
        of pipeline stages:</p>
        <div class="dag-visualization">
            <pre>{html.escape(dag_visualization)}</pre>
        </div>
        <h2>Execution Order</h2>
        <p><strong>{" → ".join(execution_order)}</strong></p>
    </section>
    """


def generate_methods_section(*, records, pipeline_name):
    """Generate methods section with workflow citations.

    Parameters
    ----------
    records : list of dict
        Stage records.
    pipeline_name : str
        Name of the pipeline.
    """
    descriptions = []
    for record in records:
        cli_name = record.get("cli")
        if cli_name not in cli_flows:
            continue
        module_name, class_name = cli_flows[cli_name]
        try:
            doc = getattr(importlib.import_module(module_name), class_name).__doc__
            first_line = doc.strip().split("\n")[0] if doc else cli_name
        except Exception:
            first_line = cli_name
        descriptions.append(
            f"<li><strong>{record['name']}</strong>: {first_line} ({cli_name})</li>"
        )

    steps = "<ol>" + "\n".join(descriptions) + "</ol>" if descriptions else ""
    return f"""
    <section id="methods">
        <h1>Methods</h1>
        <div class="boilerplate">
            <p>Results included in this report were generated using DIPY Auto
            (Diffusion Imaging in Python), a comprehensive pipeline for diffusion
            MRI data processing and analysis.</p>
            <h2>Pipeline: {pipeline_name}</h2>
            <p>The following processing steps were applied:</p>
            {steps}
            <h2>References</h2>
            <p>Please cite the following references when using DIPY Auto:</p>
            <div class="citation">Garyfallidis E, Brett M, Amirbekian B, Rokem A,
van der Walt S, Descoteaux M, Nimmo-Smith I, and Dipy Contributors (2014).
DIPY, a library for the analysis of diffusion MRI data.
Frontiers in Neuroinformatics, vol.8, no.8.
https://doi.org/10.3389/fninf.2014.00008</div>
        </div>
    </section>
    """


def generate_html_report(
    *, config, config_file_path, execution_info, output_path, reports_dir
):
    """Generate the final report combining every stage QC page.

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
    reports_dir : str
        Directory holding the stage pages and their JSON records.

    Returns
    -------
    str
        Path of the report.
    """
    logger.info("Generating final HTML report from the stage QC records...")
    pipeline_name = config.get("General", {}).get("name", "Pipeline")
    stage_names = execution_info.get("execution_order", [])
    records = load_stage_records(reports_dir=reports_dir, stage_names=stage_names)

    html_dir = os.path.dirname(os.path.abspath(output_path))
    href_prefix = os.path.relpath(reports_dir, html_dir) + "/"
    stages = [
        (index, record)
        for index, record in enumerate(records, start=1)
        if record is not None
    ]

    content = (
        generate_summary_section(
            config=config, execution_info=execution_info, records=records
        )
        + generate_config_section(config=config, config_file_path=config_file_path)
        + generate_pipeline_section(
            dag_visualization=execution_info.get("dag_visualization", ""),
            execution_order=stage_names,
        )
        + _overview_section(
            stage_names=stage_names, records=records, href_prefix=href_prefix
        )
        + '<section id="results"><h1>Stage Results</h1>'
        + "".join(
            render_stage_section(
                record=record,
                number=index,
                anchor=stage_basename(index=index, stage_name=record["name"]),
                page_dir=html_dir,
            )
            for index, record in stages
        )
        + "</section>"
        + generate_methods_section(
            records=[r for _, r in stages], pipeline_name=pipeline_name
        )
    )

    sections = [
        ("#summary", "Summary"),
        ("#configuration", "Configuration"),
        ("#pipeline", "Pipeline"),
        ("#overview", "Overview"),
    ]
    sidebar = (
        '<div class="sidebar-section-label">Report</div>'
        + _links(sections)
        + '<div class="sidebar-section-label">Stages</div>'
        + _links(
            [
                (f"#{stage_basename(index=i, stage_name=r['name'])}", r["name"])
                for i, r in stages
            ],
            css="sidebar-link sidebar-sublink",
        )
        + '<div class="sidebar-section-label">Info</div>'
        + _links([("#methods", "Methods")])
    )
    page = _page(
        title=f"DIPY Auto Pipeline Report - {pipeline_name}",
        subtitle=f"DIPY Auto Pipeline Report — {pipeline_name}",
        nav=sections + [("#methods", "Methods")],
        sidebar=sidebar,
        content=content,
    )
    return _write(path=output_path, text=page)


# =============================================================================
# Batch report
# =============================================================================


def _batch_row(*, result, page_dir):
    """Table row of one acquisition of a batch.

    Parameters
    ----------
    result : dict
        Record produced for one acquisition.
    page_dir : str
        Directory of the page being written, for relative links.

    Returns
    -------
    str
        HTML fragment.
    """
    status = "PASS" if result.get("status") == "completed" else "FAIL"
    report_path = result.get("report")
    link = ""

    if report_path and os.path.exists(report_path):
        link = f'<a href="{os.path.relpath(report_path, page_dir)}">open</a>'
    else:
        index = os.path.join(result.get("out_dir", ""), "reports", "index.html")
        if os.path.exists(index):
            link = f'<a href="{os.path.relpath(index, page_dir)}">open</a>'

    detail = result.get("error") or ""
    if result.get("failed_stage"):
        detail = f"{result['failed_stage']}: {detail}"

    return (
        f"<tr><td>{html.escape(result.get('label', ''))}</td>"
        f"<td>{html.escape(result.get('subject', ''))}</td>"
        f"<td>{html.escape(result.get('session') or '')}</td>"
        f"<td>{_badge(status)}</td>"
        f"<td>{result.get('duration', 0):.1f}s</td>"
        f"<td>{html.escape(detail)}</td>"
        f"<td>{link}</td></tr>"
    )


def generate_batch_report(*, results, output_path, pipeline_name, bids_dir):
    """Write the HTML summary of a pipeline run over a BIDS dataset.

    Parameters
    ----------
    results : list of dict
        One record per acquisition, as produced by the batch loop.
    output_path : str
        Path where the HTML summary should be saved.
    pipeline_name : str
        Name of the pipeline that was run.
    bids_dir : str
        Root of the source BIDS dataset.

    Returns
    -------
    str
        Path of the summary page.
    """
    page_dir = os.path.dirname(os.path.abspath(output_path))
    succeeded = [r for r in results if r.get("status") == "completed"]
    failed = [r for r in results if r.get("status") != "completed"]
    total_time = sum(r.get("duration", 0) for r in results)

    summary = f"""
    <section id="summary">
        <h1>Summary</h1>
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-label">Acquisitions</div>
                <div class="stat-value">{len(results)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Succeeded</div>
                <div class="stat-value">{len(succeeded)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Failed</div>
                <div class="stat-value">{len(failed)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Duration</div>
                <div class="stat-value">{total_time:.0f}
                  <span style="font-size:14px;font-weight:500;
                    color:#6b7280;margin-left:4px">s</span>
                </div>
            </div>
        </div>
        <div class="info-grid">
            <div class="info-card">
                <strong>Pipeline</strong>
                <p>{html.escape(pipeline_name)}</p>
            </div>
            <div class="info-card">
                <strong>BIDS Dataset</strong>
                <p><code>{html.escape(bids_dir)}</code></p>
            </div>
            <div class="info-card">
                <strong>Derivatives</strong>
                <p><code>{html.escape(page_dir)}</code></p>
            </div>
        </div>
    </section>
    """

    rows = "".join(_batch_row(result=result, page_dir=page_dir) for result in results)
    table = f"""
    <section id="runs">
        <h1>Acquisitions</h1>
        <table>
            <thead><tr><th>Run</th><th>Subject</th><th>Session</th>
            <th>Status</th><th>Duration</th><th>Detail</th>
            <th>Report</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </section>
    """

    sections = [("#summary", "Summary"), ("#runs", "Acquisitions")]
    sidebar = (
        '<div class="sidebar-section-label">Batch</div>'
        + _links(sections)
        + '<div class="sidebar-section-label">Acquisitions</div>'
        + _links(
            [
                (
                    os.path.relpath(
                        os.path.join(r.get("out_dir", ""), "reports", "index.html"),
                        page_dir,
                    ),
                    r.get("label", ""),
                )
                for r in results
            ],
            css="sidebar-link sidebar-sublink",
        )
    )
    page = _page(
        title=f"DIPY Auto Batch Report - {pipeline_name}",
        subtitle=f"DIPY Auto BIDS batch — {pipeline_name}",
        nav=sections,
        sidebar=sidebar,
        content=summary + table,
    )
    return _write(path=output_path, text=page)
