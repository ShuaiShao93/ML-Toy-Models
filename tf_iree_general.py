"""
General tf saved model to IREE.

nsys profile --force-overwrite true -w true -t cublas,cuda,nvtx,osrt -s cpu -o /tmp/tf_iree_general python tf_iree_general.py
"""

import os
import time
import numpy as np
import tensorflow as tf

import iree.compiler as ireec
import iree.runtime as ireert

SAVED_MODEL_DIR = "/tmp/ilp"
MLIR_PATH = "/tmp/ilp.mlir"

saved_model = tf.saved_model.load(SAVED_MODEL_DIR)

os.system("iree-import-tf --output-format=mlir-bytecode --tf-import-type=savedmodel_v2 --tf-savedmodel-exported-names=forward " +
          SAVED_MODEL_DIR + " -o " + MLIR_PATH)
with open(MLIR_PATH, "r") as f:
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

concrete_function = saved_model.signatures['serving_default']
arg_specs, kwarg_specs = concrete_function.structured_input_signature
input_tensors = []
for arg in concrete_function._arg_keywords:
    dtype = tf.int32 if kwarg_specs[arg].dtype == tf.int64 else kwarg_specs[arg].dtype
    input_tensors.append(tf.zeros(shape=kwarg_specs[arg].shape, dtype=dtype))

# warmup
iree_input_batch = [ireert.asdevicearray(
    iree_device, input_batch.numpy()) for input_batch in input_tensors]
result = invoker.forward(*iree_input_batch)

start = time.time()
result = invoker.forward(*iree_input_batch)
numpy_result = np.asarray(result)
print("iree result", numpy_result[0][0])
print("iree time", time.time() - start)
