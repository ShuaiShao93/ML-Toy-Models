import os

import tensorflow as tf
import numpy as np
import onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

DYNAMIC_BATCH_SIZE = True
SAVED_MODEL_DIR = "/tmp/resnet50"
ONNX_MODEL_PATH = SAVED_MODEL_DIR + ".onnx"
TRT_MODEL_PATH = SAVED_MODEL_DIR + ".trt"


class ResNet50(tf.Module):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.applications.resnet50.ResNet50(
            weights="imagenet",
            include_top=True
        )

    @tf.function
    def forward(self, inputs):
        return self.model(inputs, training=False)

resnet = ResNet50()
input_batch = np.float32(np.random.rand(1, 224, 224, 3))
output = resnet.forward(input_batch).numpy()
print("tf result", output[0, 0])

# Save saved model.
batch_size = None if DYNAMIC_BATCH_SIZE else 1
tensor_specs = [tf.TensorSpec((batch_size, 224, 224, 3), tf.float32)]
call_signature = resnet.forward.get_concrete_function(*tensor_specs)

os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
print(f"Saving {SAVED_MODEL_DIR} with call signature: {call_signature}")
tf.saved_model.save(resnet, SAVED_MODEL_DIR,
                    signatures={"serving_default": call_signature})

# convert to onnx
assert os.system(
    f"python -m tf2onnx.convert --saved-model {SAVED_MODEL_DIR} --output {ONNX_MODEL_PATH}") == 0

# # Load serialized ONNX model and remove transpose
# onnx_model = onnx.load(ONNX_MODEL_PATH)
# onnx.checker.check_model(onnx_model)
# transpose_node = None
# for node in onnx_model.graph.node:
#     if node.op_type == "Transpose":
#         transpose_node = node
#         onnx_model.graph.node.remove(node)
#         print("removed transpose node", node.name)
#         break
# assert transpose_node
# for node in onnx_model.graph.node:
#     for i, input in enumerate(node.input):
#         if input.startswith(transpose_node.name):
#             node.input[i] = transpose_node.input[0]
#             print("updated input", node.name)
# onnx.save(onnx_model, ONNX_MODEL_PATH)

# convert to trt
args = [
    f"--onnx={ONNX_MODEL_PATH}",
    f"--saveEngine={TRT_MODEL_PATH}",
    "--inputIOFormats=fp32:hwc"
]
if DYNAMIC_BATCH_SIZE:
    args.append("--minShapes=inputs:1x224x224x3")
    args.append("--maxShapes=inputs:8x224x224x3")
    args.append("--optShapes=inputs:1x224x224x3")

assert os.system(f"trtexec {' '.join(args)}") == 0

# Load serialized TensorRT model, and run inference.
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
with open(TRT_MODEL_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

if DYNAMIC_BATCH_SIZE:
    context.set_binding_shape(0, input_batch.shape)

d_input = cuda.mem_alloc(1 * input_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

cuda.memcpy_htod_async(d_input, input_batch, stream)
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
trt_output = np.empty_like(output)
cuda.memcpy_dtoh_async(trt_output, d_output, stream)
stream.synchronize()
print("trt result", trt_output[0, 0])

assert np.allclose(output, trt_output, atol=1e-2)
