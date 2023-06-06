import os
import onnx

INPUT_SAVED_MODEL_DIR = "/tmp/toy"
ONNX_MODEL_PATH = INPUT_SAVED_MODEL_DIR + ".onnx"

# convert to onnx
os.system(
    f"python -m tf2onnx.convert --saved-model {INPUT_SAVED_MODEL_DIR} --output {ONNX_MODEL_PATH}")

# inspect onnx graph/node
onnx_model = onnx.load(ONNX_MODEL_PATH)
for node in onnx_model.graph.node:
    if node.name == "NODE NAME":
        print(node)

# convert to trt
os.system(f"trtexec --onnx={ONNX_MODEL_PATH} --dumpProfile")
