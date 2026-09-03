"""Fix CUDA's bundled CCCL headers so GCC will compile them.

Five ONNX Runtime builds died at 98% in the same place::

    cccl/cub/device/device_transform.cuh:44: error: global qualification of
    class name is invalid before ':' token
      struct ::cuda::proclaims_copyable_arguments<...> : ::cuda::std::true_type

A template specialization cannot be declared with a global-qualified name at
namespace scope. NVIDIA has already fixed this one: CCCL ``main`` writes the
same declaration wrapped in ``namespace cuda { ... }`` with the qualifier
dropped, and that is exactly what this produces. CUDA 13 still ships the old
form.

CUDA 13.2 contains two of these - the second, in tuning_transform.cuh, was
restructured away upstream rather than fixed in place, so it has no published
fixed form to copy; it is the identical construct and gets the identical
treatment. The two differ in shape (``template <typename T>`` vs ``template
<>``, base clause on the struct line vs. its own), so the pattern below is
written to tolerate both rather than to match one verbatim.

Wrapping in plain ``namespace cuda`` is right even though the primary template
lives in a versioned inline namespace inside it: a specialization may be
declared in the innermost enclosing non-inline namespace.

Note that --disable_contrib_ops, which would have dropped the translation
units that include these, is rejected outright by
onnxruntime_providers_tensorrt.cmake - the TensorRT execution provider needs
contrib ops for its EPContext node.

Run inside the wheel builder, not on a developer machine.
"""

import pathlib
import re
import sys

CUDA_ROOT = pathlib.Path("/usr/local/cuda")

# The specialization, and the namespace-wrapped form upstream replaced it
# with. Anchored on the template line above it, and closed on the empty body,
# so the braces matched belong to this declaration and not to what follows.
# [^{}]* spans the newlines of a multi-line base clause; there is no brace
# between the struct name and its body.
BROKEN = re.compile(
    r"^(template <[^\n]*>\n)"
    r"struct ::cuda::(proclaims_copyable_arguments<[^{}]*)"
    r"(\{\};)$",
    re.MULTILINE,
)
FIXED = (
    "namespace cuda\n"
    "{\n"
    r"\1struct \2\3"
    "\n"
    "} // namespace cuda"
)

# Both live in .cuh files; the verification sweep below covers headers
# generally, not just the two known names.
EXPECTED_OCCURRENCES = 2


def _read(header: pathlib.Path) -> str:
    """Decode as latin-1 so any byte round-trips unchanged.

    These headers are ASCII where it matters, but a stray non-UTF-8 byte
    anywhere under CUDA_ROOT must not crash the sweep.
    """
    return header.read_bytes().decode("latin-1")


def _write(header: pathlib.Path, text: str) -> None:
    header.write_bytes(text.encode("latin-1"))


def main() -> int:
    """Patch every broken specialization, then verify none is left."""
    patched = []
    total = 0
    for header in sorted(CUDA_ROOT.rglob("*.cuh")):
        text = _read(header)
        if "struct ::cuda::" not in text:
            continue
        new_text, count = BROKEN.subn(FIXED, text)
        if count:
            _write(header, new_text)
            patched.append(f"{header} ({count} occurrence(s))")
            total += count

    if not patched:
        # Loud rather than quiet: if CUDA ships this fixed, the patch should
        # be deleted, not left silently doing nothing.
        print(
            "No CCCL header needed patching. Either CUDA fixed this upstream - "
            "in which case delete this script and the step that runs it - or "
            "the declaration moved and the pattern no longer matches.",
            file=sys.stderr,
        )
        return 1

    print("patched:", *patched, sep="\n  ")

    # The postcondition is what matters, not the edit. Anything still
    # declaring one of these would fail the build hours from now.
    leftover = False
    for header in sorted(CUDA_ROOT.rglob("*.cuh")):
        text = _read(header)
        if "struct ::cuda::" in text:
            print(
                f"{header} still declares a global-qualified specialization",
                file=sys.stderr,
            )
            leftover = True
    if leftover:
        return 1

    if total != EXPECTED_OCCURRENCES:
        # Not fatal - the postcondition above already proves the headers
        # compile-clean - but a changed count means the CUDA base image moved
        # and this script deserves a fresh look.
        print(
            f"warning: patched {total} occurrence(s), expected "
            f"{EXPECTED_OCCURRENCES}; the CUDA base image may have changed.",
            file=sys.stderr,
        )

    print(f"no global-qualified specializations remain under {CUDA_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
