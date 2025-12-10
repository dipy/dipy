"""HTML report generation for dipy_auto pipeline execution.

This module generates detailed HTML reports showing pipeline configuration,
execution graph, timing information, and interactive visualizations of
neuroimaging outputs using NiiVue.
"""

from datetime import datetime
import os

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
            font-family: "Bitstream Charter", "Georgia", Times, serif;
            line-height: 1.6;
            color: #333;
            background: #f8f9fa;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}

        header {{
            background: linear-gradient(135deg, #E87722 0%, #D94A38 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        nav {{
            background: #2c3e50;
            padding: 15px 40px;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}

        nav a {{
            color: white;
            text-decoration: none;
            margin-right: 25px;
            font-weight: 500;
            transition: color 0.3s;
        }}

        nav a:hover {{
            color: #E87722;
        }}

        .content {{
            padding: 40px;
        }}

        section {{
            margin-bottom: 60px;
        }}

        h1 {{
            font-size: 2.2em;
            margin-bottom: 20px;
            padding-top: 35px;
            color: #2c3e50;
            border-bottom: 3px solid #E87722;
            padding-bottom: 10px;
        }}

        h2 {{
            font-size: 1.8em;
            margin-bottom: 15px;
            padding-top: 20px;
            color: #34495e;
        }}

        h3 {{
            font-size: 1.4em;
            margin-bottom: 10px;
            padding-top: 15px;
            color: #555;
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .info-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #E87722;
        }}

        .info-card strong {{
            display: block;
            color: #E87722;
            margin-bottom: 8px;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .info-card p {{
            color: #555;
            word-break: break-word;
        }}

        .dag-visualization {{
            background: #f8f9fa;
            padding: 30px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 20px 0;
        }}

        .dag-visualization pre {{
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.8;
            color: #2c3e50;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        thead {{
            background: #E87722;
            color: white;
        }}

        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}

        tbody tr:hover {{
            background: #f8f9fa;
        }}

        .stage-section {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 25px;
            margin: 30px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}

        .stage-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #E87722;
        }}

        .stage-title {{
            font-size: 1.5em;
            color: #2c3e50;
        }}

        .stage-time {{
            background: #E87722;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }}

        .viewer-container {{
            margin: 25px 0;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }}

        .viewer-header {{
            background: #34495e;
            color: white;
            padding: 12px 20px;
            font-weight: 500;
        }}

        canvas {{
            display: block;
            width: 100%;
            height: 500px;
            background: #000;
        }}

        .output-list {{
            margin: 20px 0;
        }}

        .output-item {{
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 3px solid #3498db;
        }}

        .output-item strong {{
            color: #3498db;
            display: block;
            margin-bottom: 5px;
        }}

        .output-item code {{
            background: #fff;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.9em;
            color: #e74c3c;
        }}

        .download-link {{
            display: inline-block;
            margin-top: 10px;
            padding: 8px 16px;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-size: 0.9em;
            transition: background 0.3s;
        }}

        .download-link:hover {{
            background: #2980b9;
        }}

        footer {{
            background: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
            margin-top: 40px;
        }}

        .error-box {{
            background: #fee;
            border: 1px solid #fcc;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
            color: #c33;
        }}

        .success-box {{
            background: #efe;
            border: 1px solid #cfc;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
            color: #3c3;
        }}

        .boilerplate {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 8px;
            margin: 20px 0;
            line-height: 1.8;
        }}

        .boilerplate p {{
            margin-bottom: 15px;
        }}

        .citation {{
            background: #fff;
            border-left: 3px solid #E87722;
            padding: 15px;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}

        @media print {{
            nav {{
                position: relative;
            }}
            .viewer-container {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p>Generated on {timestamp}</p>
        </header>

        <nav>
            <a href="#summary">Summary</a>
            <a href="#configuration">Configuration</a>
            <a href="#pipeline">Pipeline</a>
            <a href="#timing">Timing</a>
            <a href="#results">Results</a>
            <a href="#methods">Methods</a>
        </nav>

        <div class="content">
            {content}
        </div>

        <footer>
            <p>Generated by DIPY Auto Pipeline</p>
            <p>DIPY - Diffusion Imaging in Python</p>
        </footer>
    </div>

    <script src="https://unpkg.com/@niivue/niivue@latest/dist/niivue.umd.js"></script>
    <script>
        {niivue_scripts}
    </script>
</body>
</html>
"""


def generate_summary_section(config, execution_info):
    """Generate summary section with key information."""
    general = config.get("General", {})
    io_config = config.get("io", {})

    summary_html = """
    <section id="summary">
        <h1>Summary</h1>
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
                <strong>Total Stages</strong>
                <p>{num_stages}</p>
            </div>
            <div class="info-card">
                <strong>Total Duration</strong>
                <p>{total_time:.2f} seconds</p>
            </div>
            <div class="info-card">
                <strong>Output Directory</strong>
                <p><code>{out_dir}</code></p>
            </div>
            <div class="info-card">
                <strong>Status</strong>
                <p class="success-box">✓ Completed Successfully</p>
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


def generate_timing_section(stages_info):
    """Generate timing table for all stages."""
    rows = ""
    for stage in stages_info:
        rows += """
        <tr>
            <td>{name}</td>
            <td>{cli}</td>
            <td>{duration:.2f}s</td>
            <td>{status}</td>
        </tr>
        """.format(
            name=stage["name"],
            cli=stage.get("cli", "N/A"),
            duration=stage.get("duration", 0),
            status="✓ Success" if stage.get("success", True) else "✗ Failed",
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


def generate_niivue_viewer(canvas_id, file_path, viewer_title, html_dir):
    """Generate NiiVue viewer HTML and JavaScript.

    Parameters
    ----------
    canvas_id : str
        Unique canvas ID for this viewer.
    file_path : str
        Absolute path to the data file.
    viewer_title : str
        Title for the viewer.
    html_dir : str
        Directory where the HTML file is saved (for computing relative paths).
    """
    file_path_str = str(file_path)

    rel_path = os.path.relpath(file_path_str, html_dir)

    viewer_html = """
    <div class="viewer-container">
        <div class="viewer-header">{title}</div>
        <canvas id="{canvas_id}"></canvas>
    </div>
    """.format(canvas_id=canvas_id, title=viewer_title)

    viewer_script = """
    (function() {{
        var nv{canvas_id} = new niivue.Niivue({{
            isResizeCanvas: false,
            textHeight: 0.05
        }});
        nv{canvas_id}.attachTo('{canvas_id}');
        nv{canvas_id}.loadVolumes([{{
            url: '{file_path}'
        }}]);
    }})();
    """.format(canvas_id=canvas_id, file_path=rel_path)

    return viewer_html, viewer_script


def is_viewable_format(file_path):
    """Check if file is viewable in NiiVue."""
    if not file_path:
        return False

    file_path_str = str(file_path)

    if not os.path.exists(file_path_str):
        return False

    ext = os.path.splitext(file_path_str.lower())[1]
    if file_path_str.lower().endswith(".nii.gz"):
        return True
    viewable_exts = [
        ".nii",
        ".nrrd",
        ".mgh",
        ".mgz",
        ".trk",
        ".tck",
        ".vtk",
        ".stl",
        ".gii",
    ]
    return ext in viewable_exts


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
            <p><strong>Note:</strong> If the 3D viewers above show "loading..."
            indefinitely, you need to serve this report via HTTP. In your terminal,
            navigate to this report's directory and run:
            <code>python -m http.server 8000</code>, then open
            <code>http://localhost:8000/pipeline_report.html</code></p>

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


def generate_results_section(stages_info, html_dir):
    """Generate results section with viewers for each stage.

    Parameters
    ----------
    stages_info : list
        List of stage information dictionaries.
    html_dir : str
        Directory where the HTML file is saved (for computing relative paths).
    """
    results_html = '<section id="results"><h1>Stage Results</h1>'
    niivue_scripts = []

    for idx, stage in enumerate(stages_info):
        stage_name = stage["name"]
        cli = stage.get("cli", "Unknown")
        duration = stage.get("duration", 0)
        outputs = stage.get("outputs", {})

        results_html += """
        <div class="stage-section">
            <div class="stage-header">
                <div class="stage-title">Stage {num}: {name}</div>
                <div class="stage-time">{duration:.2f}s</div>
            </div>
            <p><strong>CLI Command:</strong> <code>{cli}</code></p>
        """.format(num=idx + 1, name=stage_name, duration=duration, cli=cli)

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
                    canvas_id = f"nv_stage_{idx}_{output_param}"
                    viewer_html, viewer_script = generate_niivue_viewer(
                        canvas_id,
                        output_path,
                        f"{output_param}: {os.path.basename(output_path)}",
                        html_dir,
                    )
                    results_html += viewer_html
                    niivue_scripts.append(viewer_script)

            results_html += "</div>"
        else:
            results_html += "<p><em>No outputs recorded for this stage.</em></p>"

        results_html += "</div>"

    results_html += "</section>"

    return results_html, "\n".join(niivue_scripts)


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
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pipeline_name = config.get("General", {}).get("name", "Pipeline")
    title = f"DIPY Auto Pipeline Report - {pipeline_name}"

    html_dir = os.path.dirname(os.path.abspath(output_path))

    summary = generate_summary_section(config, execution_info)
    configuration = generate_config_section(config, config_file_path)
    pipeline = generate_pipeline_section(
        execution_info.get("dag_visualization", ""),
        execution_info.get("execution_order", []),
    )
    timing = generate_timing_section(execution_info.get("stages", []))
    results, niivue_scripts = generate_results_section(
        execution_info.get("stages", []), html_dir
    )
    methods = generate_methods_section(execution_info.get("stages", []), pipeline_name)

    content = summary + configuration + pipeline + timing + results + methods

    html = HTML_TEMPLATE.format(
        title=title, timestamp=timestamp, content=content, niivue_scripts=niivue_scripts
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    return output_path
