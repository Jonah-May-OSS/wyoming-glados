"""Tests for freezing the VITS scales into an exported graph."""

import pytest

from dataset_tools.bake_scales import DEFAULT_SCALES, bake, main

onnx = pytest.importorskip(
    "onnx", reason="onnx is only installed in the WSL export venv"
)


def _model(with_scales: bool):
    """Build a minimal graph shaped like piper's export: input, lengths, scales."""
    inputs = [
        onnx.helper.make_tensor_value_info("input", onnx.TensorProto.INT64, [1, None]),
        onnx.helper.make_tensor_value_info(
            "input_lengths", onnx.TensorProto.INT64, [1]
        ),
    ]
    if with_scales:
        inputs.append(
            onnx.helper.make_tensor_value_info("scales", onnx.TensorProto.FLOAT, [3])
        )
    node = onnx.helper.make_node("Identity", ["input"], ["output"])
    out = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.INT64, [1, None]
    )
    return onnx.helper.make_model(onnx.helper.make_graph([node], "g", inputs, [out]))


def _roundtrip(tmp_path, with_scales, scales=DEFAULT_SCALES):
    src, dst = tmp_path / "in.onnx", tmp_path / "out.onnx"
    onnx.save(_model(with_scales), str(src))
    changed = bake(str(src), str(dst), scales)
    return changed, onnx.load(str(dst)).graph


class TestBake:
    def test_scales_move_from_input_to_initializer(self, tmp_path):
        changed, graph = _roundtrip(tmp_path, with_scales=True)
        assert changed is True
        assert [i.name for i in graph.input] == ["input", "input_lengths"]
        assert [i.name for i in graph.initializer] == ["scales"]

    def test_the_baked_values_are_preserved(self, tmp_path):
        _, graph = _roundtrip(tmp_path, with_scales=True, scales=(0.1, 1.5, 0.2))
        stored = onnx.numpy_helper.to_array(graph.initializer[0])
        assert stored.tolist() == pytest.approx([0.1, 1.5, 0.2])
        assert stored.dtype == "float32"

    def test_rerunning_on_a_baked_model_is_a_noop(self, tmp_path):
        """Export may be re-run; a missing scales input is not an error."""
        changed, graph = _roundtrip(tmp_path, with_scales=False)
        assert changed is False
        assert not graph.initializer

    def test_the_result_is_a_valid_graph(self, tmp_path):
        _, graph = _roundtrip(tmp_path, with_scales=True)
        onnx.checker.check_graph(graph)


class TestMain:
    def test_wrong_argument_count_is_rejected(self):
        assert main(["only-one"]) == 2

    def test_defaults_are_used_when_no_scales_are_given(self, tmp_path):
        src, dst = tmp_path / "in.onnx", tmp_path / "out.onnx"
        onnx.save(_model(True), str(src))
        assert main([str(src), str(dst)]) == 0

        stored = onnx.numpy_helper.to_array(onnx.load(str(dst)).graph.initializer[0])
        assert stored.tolist() == pytest.approx(list(DEFAULT_SCALES))
