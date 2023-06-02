import os

INPUT_SAVED_MODEL_DIR = "/tmp/toy"
ONNX_MODEL_PATH = INPUT_SAVED_MODEL_DIR + ".onnx"


os.system(
    f"python -m tf2onnx.convert --saved-model {INPUT_SAVED_MODEL_DIR} --output {ONNX_MODEL_PATH}")

# os.system(f"trtexec --onnx={ONNX_MODEL_PATH} --dumpProfile")
