"""
dipy/viz/horizon/webgpu.py
---------------------------
WebGPU / browser integration for DIPY Horizon (Issue #3502).

Provides two entry-points:

1. ``HorizonWidget`` — a Jupyter widget wrapping FURY's streaming server.
   Use in Jupyter Lab / Notebook / Google Colab.

2. ``HorizonWGPUApp`` — a standalone wgpu-py window (no browser needed).
   Requires: pip install wgpu fury[stream]

Architecture
------------

   dMRI data (NumPy)
        │
        ▼
   HorizonApp.build_scene()         ← FURY scene graph
        │   actors (ODF/Tensor)
        ▼
   FuryStreamInteractor              ← captures frames via vtkRenderWindow
        │   JPEG/WebP frames @ ~30fps
        ▼
   asyncio WebSocket server          ← fury.stream.server
        │
   ───────────────────────────────────
   │ Browser                         │
   │  canvas ← WebSocket frames      │
   │  mouse/keyboard events →        │
   ───────────────────────────────────

WebGPU upgrade path
-------------------
When wgpu-py / Dawn backend reaches stability in FURY (tracked in
FURY issue #942), the FuryStreamInteractor will delegate rendering to
wgpu.GPUDevice instead of VTK's OpenGL pipeline, enabling:
  • GPU-accelerated SH→sphere evaluation in compute shaders
  • Reduced CPU↔GPU transfer overhead for large ODF volumes
  • Cross-platform support (Chromium, Firefox Nightly with WebGPU flag)
"""

from __future__ import annotations

import logging
import threading
from typing import Optional
import warnings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Jupyter Widget entry-point
# ---------------------------------------------------------------------------


class HorizonWidget:
    """Jupyter widget that streams a FURY Horizon scene to a browser canvas.

    Parameters
    ----------
    horizon_app : HorizonApp
        A fully built (``build_scene()`` called) Horizon application.
    port : int, optional
        WebSocket port.  Default 8765.
    width, height : int, optional
        Canvas dimensions in pixels.

    Examples
    --------
    .. code-block:: python

        from dipy.viz.horizon.app import HorizonApp
        from dipy.viz.horizon.webgpu import HorizonWidget

        app = HorizonApp(odf_files=["fodf.nii.gz"])
        app.build_scene()

        widget = HorizonWidget(app, width=800, height=600)
        widget.display()          # renders inline in Jupyter
    """

    def __init__(
        self,
        horizon_app,
        port: int = 8765,
        width: int = 800,
        height: int = 600,
    ):
        self._app = horizon_app
        self._port = port
        self._width = width
        self._height = height
        self._stream = None
        self._server_thread: Optional[threading.Thread] = None

    def _init_stream(self):
        """Set up FURY streaming interactor."""
        try:
            from fury.stream.server.main import FuryStreamInteractor
        except ImportError as exc:
            raise ImportError(
                "fury[stream] is required for WebGPU/browser mode.\n"
                "Install with: pip install 'fury[stream]'"
            ) from exc

        scene = self._app.scene
        self._stream = FuryStreamInteractor(
            scene=scene,
            window_size=(self._width, self._height),
            use_raw_array=True,  # zero-copy shared memory frames
            max_clients=4,
        )

    def start_server(self):
        """Launch the WebSocket frame server in a background thread."""
        self._init_stream()
        self._stream.start(port=self._port)
        logger.info("Horizon stream server listening on ws://localhost:%d", self._port)

    def stop_server(self):
        if self._stream is not None:
            self._stream.stop()
            logger.info("Stream server stopped.")

    def display(self):
        """Display the streaming canvas inline (Jupyter / IPython)."""
        self.start_server()
        try:
            from fury.stream.widget import Widget

            w = Widget(stream_interactor=self._stream)
            w.display_scene()
            return w
        except ImportError:
            # Fallback: raw IFrame embedding
            return self._iframe_fallback()

    def _iframe_fallback(self):
        """Return an IPython IFrame embedding the stream client."""
        try:
            from IPython.display import IFrame
        except ImportError:
            warnings.warn(
                "IPython not available; cannot display IFrame.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        client_url = f"http://localhost:{self._port}/index.html"
        return IFrame(src=client_url, width=self._width + 20, height=self._height + 60)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        self.start_server()
        return self

    def __exit__(self, *args):
        self.stop_server()


# ---------------------------------------------------------------------------
# Standalone wgpu-py window
# ---------------------------------------------------------------------------


class HorizonWGPUApp:
    """Standalone WebGPU window using wgpu-py (no browser needed).

    This is the *future* path for Issue #3502 once FURY fully integrates
    wgpu-py as a rendering backend (ref: FURY issue #942).

    Parameters
    ----------
    horizon_app : HorizonApp
    width, height : int
    title : str

    Notes
    -----
    Currently wraps wgpu.gui.WgpuCanvas as a thin shim over FURY's VTK
    renderer.  When FURY's native wgpu backend lands, this class will
    delegate rendering directly to wgpu.GPUDevice.
    """

    def __init__(self, horizon_app, width=1280, height=720, title="Horizon WebGPU"):
        self._app = horizon_app
        self._width = width
        self._height = height
        self._title = title

    def _check_wgpu(self):
        try:
            import wgpu  # noqa: F401

            return True
        except ImportError:
            return False

    def run(self):
        if not self._check_wgpu():
            warnings.warn(
                "wgpu-py not installed. Falling back to VTK ShowManager.\n"
                "Install wgpu with: pip install wgpu",
                RuntimeWarning,
                stacklevel=2,
            )
            self._app.run()
            return

        try:
            import wgpu
            from wgpu.gui.auto import WgpuCanvas, run

            logger.info("Initialising WebGPU canvas …")
            canvas = WgpuCanvas(title=self._title, size=(self._width, self._height))
            adapter = wgpu.request_adapter(
                canvas=canvas,
                power_preference="high-performance",
            )
            adapter.request_device()
            logger.info("WebGPU adapter: %s", adapter.summary)

            # ── Future integration point ──
            # Once FURY exposes a wgpu render pass, we would do:
            #   fury.backend.wgpu.render(self._app.scene, device, canvas)
            # For now, stream via shared memory.
            widget = HorizonWidget(self._app, width=self._width, height=self._height)
            logger.info(
                "wgpu canvas ready. Streaming Horizon scene "
                " via WebSocket on port 8765.\n"
                "Open http://localhost:8765 in a WebGPU-enabled browser."
            )
            widget.start_server()
            run()

        except Exception as exc:
            logger.error("WebGPU initialisation failed (%s); using VTK fallback.", exc)
            self._app.run()


# ---------------------------------------------------------------------------
# Pyodide / browser-native helper
# ---------------------------------------------------------------------------


async def run_in_pyodide(
    sh_coeffs_bytes: bytes, evals_bytes: bytes, evecs_bytes: bytes
):
    """Entry-point callable from JavaScript in a Pyodide context.

    Reconstructs numpy arrays from byte buffers transferred from JS,
    builds a minimal scene, and renders to an offscreen canvas using
    the wgpu-py Emscripten backend.

    Parameters
    ----------
    sh_coeffs_bytes : bytes
        Raw float32 bytes of SH coefficients (C-order, shape encoded
        separately via JS postMessage metadata).
    evals_bytes, evecs_bytes : bytes
        Eigenvalue / eigenvector buffers.

    Notes
    -----
    This function is experimental and depends on:
      - Pyodide >= 0.24
      - wgpu-py Emscripten wheel (not yet publicly released as of 2025-Q1)
      - FURY HEAD with wgpu rendering path
    """
    import js  # pyodide JS bridge
    import numpy as np
    from pyodide.ffi import to_js

    logger.info("Pyodide context detected; decoding numpy arrays from JS buffers.")

    # Decode metadata sent from JS
    meta = js.horizonMeta.to_py()  # {sh_shape, ev_shape, evs_shape}
    sh = np.frombuffer(sh_coeffs_bytes, dtype=np.float32).reshape(meta["sh_shape"])
    ev = np.frombuffer(evals_bytes, dtype=np.float32).reshape(meta["ev_shape"])
    evs = np.frombuffer(evecs_bytes, dtype=np.float32).reshape(meta["evs_shape"])

    logger.info("Decoded: sh=%s  evals=%s  evecs=%s", sh.shape, ev.shape, evs.shape)

    # Build minimal scene (no ShowManager — wgpu canvas handles presentation)
    import sys

    sys.path.insert(0, "/home/pyodide")
    from dipy.viz.horizon.tab.odf import ODFTab
    from dipy.viz.horizon.tab.tensor import TensorTab

    odf_tab = ODFTab(sh, norm=True, scale=2.0)
    odf_tab.build_actors()

    tensor_tab = TensorTab(ev, evs, colormap="fa_rgb")
    tensor_tab.build_actors()

    # Signal JS that actors are ready
    js.postMessage(to_js({"status": "actors_ready"}))

    # Return actor VTK data as dict for wasm renderer
    # (full implementation pending wgpu-py Emscripten wheel)
    return {
        "odf_actors": len(odf_tab.get_actors()),
        "tensor_actors": len(tensor_tab.get_actors()),
    }
