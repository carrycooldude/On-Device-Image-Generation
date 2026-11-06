"""Rebuild Qualcomm AI Hub precompiled_qnn_onnx wrapper ONNX files for ONNX Runtime Android (Java API):
1. Strips the '_q' suffix on EPContext node inputs/outputs and removes redundant DequantizeLinear->QuantizeLinear pairs
   so the EPContext node matches the exact tensor names inside the QNN context binaries (.bin).
2. Wraps UINT16 graph inputs/outputs with Cast(INT16 -> UINT16) on input and Cast(UINT16 -> INT16) on output,
   exposing '<name>_i16' (INT16) at the graph boundary since ONNX Runtime's Java API does not expose OnnxJavaType.UINT16.
"""
import os
import sys
import onnx
from onnx import TensorProto, helper

U16, I16 = TensorProto.UINT16, TensorProto.INT16


def patch_onnx_wrapper(src_path: str, dst_path: str):
    m = onnx.load(src_path, load_external_data=False)
    g = m.graph

    ep_nodes = [n for n in g.node if n.op_type == "EPContext"]
    assert len(ep_nodes) == 1, f"Expected 1 EPContext node in {src_path}, found {len(ep_nodes)}"
    ep = ep_nodes[0]

    in_map = {inp.name.removesuffix("_i16"): inp for inp in g.input}
    out_map = {out.name.removesuffix("_i16"): out for out in g.output}

    new_inputs, new_outputs, new_value_info, pre_nodes, post_nodes = [], [], [], [], []

    for idx, ep_in in enumerate(ep.input):
        base = ep_in.removesuffix("_q")
        ep.input[idx] = base
        orig_vi = in_map[base]
        elem_type = orig_vi.type.tensor_type.elem_type
        shape = orig_vi.type.tensor_type.shape

        if elem_type in (U16, I16):
            ext_name = base + "_i16"
            ext_vi = helper.make_tensor_value_info(ext_name, I16, None)
            ext_vi.type.tensor_type.shape.CopyFrom(shape)
            new_inputs.append(ext_vi)

            inner_vi = helper.make_tensor_value_info(base, U16, None)
            inner_vi.type.tensor_type.shape.CopyFrom(shape)
            new_value_info.append(inner_vi)

            pre_nodes.append(helper.make_node("Cast", [ext_name], [base], to=U16, name=base + "_cast_in"))
        else:
            new_inputs.append(orig_vi)

    for idx, ep_out in enumerate(ep.output):
        base = ep_out.removesuffix("_q")
        ep.output[idx] = base
        orig_vi = out_map[base]
        elem_type = orig_vi.type.tensor_type.elem_type
        shape = orig_vi.type.tensor_type.shape

        if elem_type in (U16, I16):
            ext_name = base + "_i16"
            ext_vi = helper.make_tensor_value_info(ext_name, I16, None)
            ext_vi.type.tensor_type.shape.CopyFrom(shape)
            new_outputs.append(ext_vi)

            inner_vi = helper.make_tensor_value_info(base, U16, None)
            inner_vi.type.tensor_type.shape.CopyFrom(shape)
            new_value_info.append(inner_vi)

            post_nodes.append(helper.make_node("Cast", [base], [ext_name], to=I16, name=base + "_cast_out"))
        else:
            new_outputs.append(orig_vi)

    del g.node[:]
    g.node.extend(pre_nodes + [ep] + post_nodes)
    del g.initializer[:]
    del g.input[:]
    g.input.extend(new_inputs)
    del g.output[:]
    g.output.extend(new_outputs)
    del g.value_info[:]
    g.value_info.extend(new_value_info)

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    onnx.save(m, dst_path)
    print(f"Patched {os.path.basename(dst_path)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/patch_qnn_onnx_wrappers.py <model_dir> [output_dir]")
        sys.exit(1)
    src_dir = sys.argv[1]
    dst_dir = sys.argv[2] if len(sys.argv) > 2 else src_dir
    for fname in os.listdir(src_dir):
        if fname.endswith(".onnx"):
            patch_onnx_wrapper(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
