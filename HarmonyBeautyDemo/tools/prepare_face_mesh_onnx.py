"""Make the Apache-2.0 MediaPipe Face Mesh ONNX compatible with MSLite 2.0."""

from pathlib import Path

import onnx
from onnx import helper, numpy_helper
from onnxsim import simplify


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "npu_face_mesh" / "face_mesh_192.onnx"
OUTPUT = ROOT / "models" / "npu_face_mesh" / "face_mesh_192_mslite.onnx"


def replace_prelu_with_relu(model: onnx.ModelProto) -> None:
    for index, node in enumerate(model.graph.node):
        if node.op_type == "PRelu":
            model.graph.node[index].CopyFrom(
                helper.make_node("Relu", [node.input[0]], [node.output[0]], name=f"{node.name}_npu_relu")
            )


def fuse_spatial_pads_into_convs(model: onnx.ModelProto) -> None:
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    removed: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.op_type != "Pad" or len(node.input) < 2:
            continue
        pads_tensor = initializers.get(node.input[1])
        if pads_tensor is None:
            continue
        pads = numpy_helper.to_array(pads_tensor).astype(int).tolist()
        next_nodes = consumers.get(node.output[0], [])
        if len(pads) != 8 or pads[:2] != [0, 0] or pads[4:6] != [0, 0]:
            continue
        if not next_nodes or any(next_node.op_type != "Conv" for next_node in next_nodes):
            continue
        if len(node.input) > 2 and node.input[2]:
            value_tensor = initializers.get(node.input[2])
            if value_tensor is None or float(numpy_helper.to_array(value_tensor).reshape(-1)[0]) != 0.0:
                continue

        spatial_pads = [pads[2], pads[3], pads[6], pads[7]]
        for conv in next_nodes:
            existing = next((attribute for attribute in conv.attribute if attribute.name == "pads"), None)
            if existing is not None and list(existing.ints) != [0, 0, 0, 0]:
                raise RuntimeError(f"Cannot merge non-zero Conv pads for {conv.name}")
            kept_attributes = [
                attribute for attribute in conv.attribute if attribute.name not in {"pads", "auto_pad"}
            ]
            del conv.attribute[:]
            conv.attribute.extend(kept_attributes)
            conv.attribute.append(helper.make_attribute("pads", spatial_pads))
            rewritten_inputs = [
                node.input[0] if input_name == node.output[0] else input_name for input_name in conv.input
            ]
            del conv.input[:]
            conv.input.extend(rewritten_inputs)
        removed.append(node)

    for node in removed:
        model.graph.node.remove(node)


def main() -> None:
    model = onnx.load(SOURCE)
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1

    for node in model.graph.node:
        if node.op_type == "MaxPool":
            kept_attributes = [attribute for attribute in node.attribute if attribute.name != "dilations"]
            del node.attribute[:]
            node.attribute.extend(kept_attributes)
    # This Kirin NNRT backend either miscomputes channel-wise PReLU or rejects
    # its standards-compliant NCHW slope shape. ReLU preserves a usable mesh and
    # keeps the graph fully on NPU.
    replace_prelu_with_relu(model)

    model, simplified = simplify(model, overwrite_input_shapes={model.graph.input[0].name: [1, 3, 192, 192]})
    if not simplified:
        raise RuntimeError("ONNX simplification validation failed")
    fuse_spatial_pads_into_convs(model)
    dynamic_pad_inputs = [
        node.name for node in model.graph.node
        if node.op_type == "Pad" and len(node.input) > 1 and
        node.input[1] not in {initializer.name for initializer in model.graph.initializer}
    ]
    if dynamic_pad_inputs:
        raise RuntimeError(f"Pad inputs were not constant-folded: {dynamic_pad_inputs}")

    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, OUTPUT)

    input_shape = [dimension.dim_value for dimension in model.graph.input[0].type.tensor_type.shape.dim]
    output_shapes = {
        output.name: [dimension.dim_value for dimension in output.type.tensor_type.shape.dim]
        for output in model.graph.output
    }
    print(f"wrote {OUTPUT}")
    print(f"input={input_shape} outputs={output_shapes}")


if __name__ == "__main__":
    main()
