"""Generate Python gRPC stubs from every .proto file under proto/.

The proto tree mirrors the package path (proto/dsearch/proto/*.proto), so the
generated modules import each other as `dsearch.proto.<name>_pb2` (see D-001).
Generated files are gitignored; never edit them by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = ROOT / "proto"
OUT_DIR = ROOT / "src"


def main() -> int:
    from grpc_tools import protoc  # dev dependency, only needed at build time
    import grpc_tools

    protos = sorted(str(p.relative_to(PROTO_DIR)) for p in PROTO_DIR.rglob("*.proto"))
    if not protos:
        print(f"no .proto files under {PROTO_DIR}", file=sys.stderr)
        return 1

    well_known = Path(grpc_tools.__file__).parent / "_proto"
    args = [
        "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"-I{well_known}",
        f"--python_out={OUT_DIR}",
        f"--pyi_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        *protos,
    ]
    code = protoc.main(args)
    if code != 0:
        print(f"protoc failed with exit code {code}", file=sys.stderr)
        return code
    print(f"generated stubs for {', '.join(protos)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
