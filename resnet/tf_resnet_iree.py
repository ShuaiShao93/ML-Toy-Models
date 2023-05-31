import os
import time

import tensorflow as tf
import numpy as np

import iree.compiler as ireec
import iree.runtime as ireert
from iree.compiler import tf as tfc

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

saved_model_path = "/tmp/resnet50"
os.makedirs(saved_model_path, exist_ok=True)
print(f"Saving {saved_model_path} with call signature: {call_signature}")
tf.saved_model.save(resnet, saved_model_path,
                    signatures={"serving_default": call_signature})

# mlir = tfc.compile_saved_model(saved_model_path, import_type=tfc.ImportType.V2, import_only=True, exported_names="forward", saved_model_tags="serving")
# print(mlir)

os.system("iree-import-tf --output-format=mlir-bytecode --tf-import-type=savedmodel_v2 --tf-savedmodel-exported-names=forward /tmp/resnet50 -o /tmp/resnet50.mlir")
with open("/tmp/resnet50.mlir", "r") as f:
    mlir = f.read()

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
