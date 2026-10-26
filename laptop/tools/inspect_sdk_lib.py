"""Tell whether a Windows .lib is an import library (which DLL it needs) or a static library.

    python tools/inspect_sdk_lib.py "C:\\path\\to\\glasses.lib"

Prints the size, the DLL names referenced (an import library names its DLL in every member), how
many exported functions look like the VITURE C API, and a verdict. Reads the file only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

API_PREFIXES = (b"xr_device_provider", b"get_gl_pose", b"register_callbacks", b"xr_camera")


def inspect(data: bytes) -> dict:
    if not data.startswith(b"!<arch>\n"):
        return {"verdict": "not a COFF archive (.lib)"}
    dlls = sorted({m.decode(errors="replace") for m in re.findall(rb"[\w.\-]+\.dll", data, re.I)})
    imports = len(re.findall(rb"__imp_", data))
    objs = len(re.findall(rb"\.obj/?", data, re.I))
    api = sorted({m.decode() for m in re.findall(rb"(?:%s)\w*" % b"|".join(API_PREFIXES), data)})
    if dlls and imports and not objs:
        verdict = f"import library -> needs {', '.join(dlls)} at runtime"
    elif objs and not imports:
        verdict = "static library -> no DLL exists; PointView needs a small C++ wrapper DLL"
    else:
        verdict = "mixed/unclear (send this output to the developer)"
    return {
        "dlls": dlls,
        "__imp_ symbols": imports,
        ".obj members": objs,
        "api": api,
        "verdict": verdict,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    path = Path(argv[1])
    data = path.read_bytes()
    print(f"{path} ({len(data):,} bytes)")
    for k, v in inspect(data).items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
