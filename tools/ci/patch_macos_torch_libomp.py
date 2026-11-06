"""Link macOS source builds and repaired wheels to a shared OpenMP runtime.

The build-only libomp copy gives source extensions an @rpath dependency. At
runtime, they prefer PyTorch's libomp and fall back to the local installation.
After delocate-wheel, the wheel repair substitutes its bundled fallback.
"""

import base64
import csv
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


LIBOMP = "@rpath/libomp.dylib"


def make_link_stub(source, destination):
    """Give the build-only linker input an @rpath ID; keep the real dylib intact."""
    shutil.copy2(source, destination)
    subprocess.run(
        ["install_name_tool", "-id", LIBOMP, str(destination)], check=True
    )
    subprocess.run(["codesign", "--force", "--sign", "-", str(destination)], check=True)


def _output(*args):
    return subprocess.check_output(args, text=True)


def _rpaths(binary):
    lines = _output("otool", "-l", str(binary)).splitlines()
    return {
        lines[i + 2].strip().split()[1]
        for i, line in enumerate(lines[:-2])
        if line.strip() == "cmd LC_RPATH"
        and lines[i + 2].strip().startswith("path ")
    }


def _patch_extension(extension, root):
    dependencies = [
        line.strip().split(" ", 1)[0]
        for line in _output("otool", "-L", str(extension)).splitlines()[1:]
    ]
    omp_dependencies = [d for d in dependencies if d.endswith("/libomp.dylib")]
    if not omp_dependencies:
        return False

    for dependency in omp_dependencies:
        if dependency != LIBOMP:
            subprocess.run(
                ["install_name_tool", "-change", dependency, LIBOMP, str(extension)],
                check=True,
            )

    # Source builds carry absolute runpaths for their build environment.
    # Delocated wheels must search only their own relative torch/fallback paths.
    for path in _rpaths(extension):
        if path.startswith("/") and (
            path.endswith("/torch/lib") or (Path(path) / "libomp.dylib").is_file()
        ):
            subprocess.run(
                ["install_name_tool", "-delete_rpath", path, str(extension)],
                check=True,
            )

    # The extension may live in dipy/utils, dipy/align or a deeper subpackage.
    # Resolve both locations relative to *its own* directory, not the package.
    for library_dir in (root / "torch" / "lib", root / "dipy" / ".dylibs"):
        path = "@loader_path/" + os.path.relpath(library_dir, extension.parent)
        if path not in _rpaths(extension):
            subprocess.run(
                ["install_name_tool", "-add_rpath", path, str(extension)],
                check=True,
            )
    subprocess.run(["codesign", "--force", "--sign", "-", str(extension)], check=True)
    return True


def _repack(root, wheel):
    record, = (root).glob("*.dist-info/RECORD")
    with record.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            name = path.relative_to(root).as_posix()
            if path == record:
                writer.writerow((name, "", ""))
            else:
                content = path.read_bytes()
                digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest())
                writer.writerow((name, "sha256=" + digest.rstrip(b"=").decode(), len(content)))

    replacement = wheel.with_suffix(".tmp.whl")
    with ZipFile(replacement, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(root).as_posix())
    replacement.replace(wheel)


def patch_wheel(wheel):
    with TemporaryDirectory() as directory:
        root = Path(directory)
        with ZipFile(wheel) as archive:
            archive.extractall(root)
        if not (root / "dipy" / ".dylibs" / "libomp.dylib").is_file():
            raise RuntimeError("Repair with delocate-wheel before patching OpenMP")
        patched = sum(
            _patch_extension(extension, root)
            for extension in (root / "dipy").rglob("*.so")
        )
        if not patched:
            raise RuntimeError("No DIPY extension links to libomp.dylib")
        _repack(root, wheel)
        print(f"Patched {patched} OpenMP extensions in {wheel}")


if __name__ == "__main__":
    if sys.platform != "darwin":
        raise SystemExit("This script requires macOS")
    if len(sys.argv) == 4 and sys.argv[1] == "--link-stub":
        make_link_stub(Path(sys.argv[2]), Path(sys.argv[3]))
    elif len(sys.argv) > 1 and sys.argv[1] != "--link-stub":
        for filename in sys.argv[1:]:
            patch_wheel(Path(filename))
    else:
        raise SystemExit(
            "Usage: patch_macos_torch_libomp.py --link-stub INPUT OUTPUT | WHEEL [...]"
        )
