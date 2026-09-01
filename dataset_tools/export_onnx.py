"""Export a Piper checkpoint to ONNX using the legacy TorchScript exporter.

torch 2.13 defaults torch.onnx.export to the dynamo path, which cannot trace
VITS: the spline flow in transforms.rational_quadratic_spline runs a
data-dependent `assert (discriminant >= 0).all()`, and dynamo raises
GuardOnDataDependentSymNode on it. The TorchScript exporter traces straight
through the assert, so it is forced here.

Also allowlists pathlib globals, since PyTorch 2.6+ defaults torch.load to
weights_only=True and the checkpoints pickle a PosixPath.
"""

import pathlib

import torch.onnx
import torch.serialization

torch.serialization.add_safe_globals(
    [pathlib.PosixPath, pathlib.WindowsPath, pathlib.PurePosixPath, pathlib.PurePath]
)

_original_export = torch.onnx.export


def _export_without_dynamo(*args, **kwargs):
    kwargs["dynamo"] = False
    return _original_export(*args, **kwargs)


torch.onnx.export = _export_without_dynamo

from piper.train.export_onnx import main  # noqa: E402

main()
