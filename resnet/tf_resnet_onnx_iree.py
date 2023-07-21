import os
import time

import onnx
import tensorflow as tf
import numpy as np

import iree.compiler as ireec
import iree.runtime as ireert

SAVED_MODEL_PATH = "/tmp/resnet50"
ONNX_MODEL_PATH = SAVED_MODEL_PATH + ".onnx"
MHLO_PATH = SAVED_MODEL_PATH + ".mhlo.mlir"
VMFB_PATH = SAVED_MODEL_PATH + ".vmfb"

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
print("tf result", resnet.forward(input_batch)[0, 0])

# Save saved model.
tensor_specs = [tf.TensorSpec((1, 224, 224, 3), tf.float32)]
call_signature = resnet.forward.get_concrete_function(*tensor_specs)

os.makedirs(SAVED_MODEL_PATH, exist_ok=True)
print(f"Saving {SAVED_MODEL_PATH} with call signature: {call_signature}")
tf.saved_model.save(resnet, SAVED_MODEL_PATH,
                    signatures={"serving_default": call_signature})

# convert to onnx
assert os.system(
    f"python -m tf2onnx.convert --saved-model {SAVED_MODEL_PATH} --output {ONNX_MODEL_PATH}") == 0
print(f"Saved onnx to {ONNX_MODEL_PATH}")

# inspect onnx graph/node
# onnx_model = onnx.load(ONNX_MODEL_PATH)
# for node in onnx_model.graph.node:
#     if node.op_type == "Conv":
#         print(node)

# onnx to mhlo
ONNX_FRONTEND_PATH = "~/byteir/frontends/onnx-frontend/build/onnx-frontend/src/onnx-frontend"
assert os.system(f"{ONNX_FRONTEND_PATH} {ONNX_MODEL_PATH} -batch-size=1 -o {MHLO_PATH}") == 0
print(f"Saved mhlo to {MHLO_PATH}")

mlir = open(MHLO_PATH, "r").read()
device = "cuda"
iree_input_type = "mhlo"
flatbuffer = ireec.compile_str(mlir,
                              target_backends=[device],
                              input_type=iree_input_type,
                              extra_args=[
                                "--iree-hal-cuda-llvm-target-arch=sm_75",
                                # "--mlir-print-ir-after=iree-flow-form-dispatch-workgroups", "--mlir-elide-elementsattrs-if-larger=8",
                                # "--iree-flow-dump-dispatch-graph", "--iree-flow-dump-dispatch-graph-output-file=foo.dot"
                              ])

iree_device = ireert.get_device(device)
config = ireert.Config(device=iree_device)
ctx = ireert.SystemContext(config=config)
vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, flatbuffer)
ctx.add_vm_module(vm_module)
invoker = ctx.modules.module

# warmup
iree_input_batch = ireert.asdevicearray(iree_device, input_batch)
result = invoker.forward(iree_input_batch)

start = time.time()
result = invoker.forward(iree_input_batch)
numpy_result = np.asarray(result)
print("iree result", numpy_result[0][0])
print("iree time", time.time() - start)

