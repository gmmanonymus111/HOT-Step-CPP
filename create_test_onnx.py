"""Create a tiny ONNX model for TRT engine smoke test."""
import numpy as np
import os

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    print("Installing onnx...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx", "-q"])
    import onnx
    from onnx import helper, TensorProto, numpy_helper

# Simple model: linear layer (matmul + add)
# Input: [batch, 4]  ->  Output: [batch, 8]
W_init = numpy_helper.from_array(
    np.random.randn(4, 8).astype(np.float32), name="W"
)
B_init = numpy_helper.from_array(
    np.random.randn(8).astype(np.float32), name="B"
)

X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8])

matmul = helper.make_node("MatMul", ["input", "W"], ["mm_out"])
add = helper.make_node("Add", ["mm_out", "B"], ["output"])
relu = helper.make_node("Relu", ["output"], ["output_relu"])

Y_relu = helper.make_tensor_value_info("output_relu", TensorProto.FLOAT, [1, 8])

graph = helper.make_graph(
    [matmul, add, relu], "smoke_test",
    [X], [Y_relu],
    initializer=[W_init, B_init]
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8

path = "test_smoke.onnx"
onnx.save(model, path)
print(f"Saved {path} ({os.path.getsize(path)} bytes)")
