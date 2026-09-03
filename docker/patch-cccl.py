"""Fix CUDA's bundled CCCL header so GCC will compile it.

Five ONNX Runtime builds died at 98% in the same place::

    cccl/cub/device/device_transform.cuh:44: error: global qualification of
    class name is invalid before ':' token
      struct ::cuda::proclaims_copyable_arguments<...> : ::cuda::std::true_type

A template specialization cannot be declared with a global-qualified name at
namespace scope, and NVIDIA has already fixed it: CCCL ``main`` writes the
same declaration wrapped in ``namespace cuda { ... }``. CUDA 13 ships the
older, broken form. This rewrites it into the shape upstream now publishes
rather than inventing a fix.

Nothing else worked. GCC 14 does not accept it either - the canary in
Dockerfile.ortwheel compiles the header standalone and passes, so it only
breaks inside ONNX Runtime's translation units. And --disable_contrib_ops,
which would have dropped the files that include it, is rejected outright by
onnxruntime_providers_tensorrt.cmake: the TensorRT execution provider needs
contrib ops for its EPContext node.

Run inside the wheel builder, not on a developer machine.
"""

import pathlib
import re
import sys

CUDA_ROOT = pathlib.Path("/usr/local/cuda")

# The specialization, and the namespace-wrapped form upstream replaced it
# with. Anchored on the template line above it so the braces below belong to
# this declaration and not to whatever follows.
BROKEN = re.compile(
    r"(template <typename T>\n)"
    r"struct ::cuda::(proclaims_copyable_arguments<[^\n]*)\n"
    r"(\{\};)"
)
FIXED = (
    "namespace cuda\n"
    "{\n"
    r"\1struct \2"
    "\n"
    r"\3"
    "\n"
    "} // namespace cuda"
)


def main() -> int:
    """Patch every copy of the header, then verify none is left broken."""
    patched = []
    for header in CUDA_ROOT.rglob("device_transform.cuh"):
        text = header.read_text()
        new_text, count = BROKEN.subn(FIXED, text)
        if count:
            header.write_text(new_text)
            patched.append(f"{header} ({count} occurrence(s))")

    if not patched:
        # Loud rather than quiet: if CUDA ships this fixed, the patch should
        # be deleted, not left silently doing nothing.
        print(
            "No CCCL header needed patching. Either CUDA fixed it upstream - "
            "in which case delete this script and the step that runs it - or "
            "the declaration moved and this no longer matches.",
            file=sys.stderr,
        )
        return 1

    print("patched:", *patched, sep="\n  ")

    # The postcondition is what matters, not the edit. Any header still
    # declaring one of these would fail the build hours from now.
    for header in CUDA_ROOT.rglob("*.cuh"):
        if "struct ::cuda::" in header.read_text():
            print(
                f"{header} still declares a global-qualified specialization",
                file=sys.stderr,
            )
            return 1

    print(f"no global-qualified specializations remain under {CUDA_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
