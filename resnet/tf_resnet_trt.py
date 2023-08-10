import os

import tensorflow as tf
import numpy as np

SAVED_MODEL_DIR = "/tmp/resnet50"
ONNX_MODEL_PATH = SAVED_MODEL_DIR + ".onnx"

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

os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
print(f"Saving {SAVED_MODEL_DIR} with call signature: {call_signature}")
tf.saved_model.save(resnet, SAVED_MODEL_DIR,
                    signatures={"serving_default": call_signature})

# convert to onnx
assert os.system(
    f"python -m tf2onnx.convert --saved-model {SAVED_MODEL_DIR} --output {ONNX_MODEL_PATH}") == 0

# convert to trt
assert os.system(f"trtexec --onnx={ONNX_MODEL_PATH} --inputIOFormats=fp32:hwc") == 0