import time

import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

INPUT_SAVED_MODEL_DIR = "/tmp/convnet"
OUTPUT_SAVED_MODEL_DIR = "/tmp/convnet_trt"

# build input fn
saved_model = tf.saved_model.load(INPUT_SAVED_MODEL_DIR)
func = saved_model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
frozen_func = convert_to_constants.convert_variables_to_constants_v2(func)


def freeze_shape(shape):
    return [dim if dim is not None else 1 for dim in shape]


def input_fn():
    yield [
        tf.random.uniform(shape=freeze_shape(input_tensor.shape),
                          maxval=10, dtype=input_tensor.dtype)
        for input_tensor in frozen_func.inputs
    ]


# convert
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=INPUT_SAVED_MODEL_DIR)
converter.convert()
converter.build(input_fn)
converter.save(OUTPUT_SAVED_MODEL_DIR)

saved_model_loaded = tf.saved_model.load(
    OUTPUT_SAVED_MODEL_DIR, tags=[tag_constants.SERVING])
graph_func = saved_model_loaded.signatures[
    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

# warmup
arg_specs, kwarg_specs = graph_func.structured_input_signature


def spec_to_tensor(spec): return tf.zeros(
    dtype=spec.dtype, shape=freeze_shape(spec.shape))


args = [spec_to_tensor(arg_spec) for arg_spec in arg_specs]
kwarg_specs = {key: spec_to_tensor(kwarg_spec)
               for key, kwarg_spec in kwarg_specs.items()}
output = graph_func(*args, **kwarg_specs)

start = time.time()
output = graph_func(*args, **kwarg_specs)
print("trt time", time.time() - start)
