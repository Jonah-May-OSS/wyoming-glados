"""Freeze the VITS inference scales into an exported ONNX graph.

`scales` is [noise_scale, length_scale, noise_w]. Because `length_scale` feeds
the output-length computation, TensorRT classifies the whole tensor as a shape
tensor - and shape tensors must be Int32/Int64, not float. It therefore refuses
the subgraph with:

    scales is a shape tensor but its data type is not allowed

and drops that partition back to the CUDA provider. Converting `scales` from a
runtime input into a constant initializer takes it out of the shape analysis
and lets TensorRT own the graph.

The cost is that speech rate is fixed at export time. That is the same tradeoff
the old ForwardTacotron path made when it baked alpha=1.0 into its engine.

Usage:
    python -m dataset_tools.bake_scales in.onnx out.onnx [noise length noise_w]
"""

from __future__ import annotations

import sys

DEFAULT_SCALES = (0.667, 1.0, 0.8)
INPUT_NAME = "scales"


def bake(src: str, dst: str, scales: tuple[float, float, float]) -> bool:
    """Rewrite `src` to `dst` with `scales` as an initializer.

    Returns True if a scales input was converted, False if the graph already
    had none - re-running on a baked model is a no-op, not an error.
    """
    import numpy as np  # pylint: disable=import-outside-toplevel
    import onnx  # pylint: disable=import-outside-toplevel
    from onnx import numpy_helper  # pylint: disable=import-outside-toplevel

    model = onnx.load(src)
    graph = model.graph

    found = [i for i in graph.input if i.name == INPUT_NAME]
    for entry in found:
        graph.input.remove(entry)
    if found:
        graph.initializer.append(
            numpy_helper.from_array(np.asarray(scales, dtype=np.float32), INPUT_NAME)
        )
    onnx.save(model, dst)
    return bool(found)


def main(argv: list[str]) -> int:
    """CLI entry point. Returns a process exit status."""
    if len(argv) not in (2, 5):
        print(__doc__, file=sys.stderr)
        return 2
    src, dst = argv[0], argv[1]
    values = tuple(float(a) for a in argv[2:]) if len(argv) == 5 else DEFAULT_SCALES
    if bake(src, dst, values):  # type: ignore[arg-type]
        print(f"  scales {list(values)} baked in as an initializer")
    else:
        print("  no scales input found; graph already baked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
