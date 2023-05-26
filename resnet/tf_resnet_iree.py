import tensorflow as tf

from iree import runtime as ireert
from iree.compiler import compile_str
from iree.compiler import tf as tfc

inputs = tf.keras.Input(shape=(32, 32, 3))
resnet = tf.keras.applications.ResNet50(weights='imagenet',
                                        include_top=True, input_tensor=inputs)

resnet.save("/tmp/resnet50")

loaded_model = tf.saved_model.load("/tmp/resnet50")
print(loaded_model.signatures.keys())

compiler_module = tfc.compile_saved_model("/tmp/resnet50", import_only=True, exported_names="serving")