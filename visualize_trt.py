# cd TensorRT/tools/experimental/trt-engine-explorer
# python3 -m pip install -e .
import os
import trex

MODEL_NAME = "resnet50"
MODEL_PATH = f"/tmp/{MODEL_NAME}.onnx"

cmd = "python3 TensorRT/tools/experimental/trt-engine-explorer/utils/process_engine.py"
cmd += f" {MODEL_PATH} /tmp/{MODEL_NAME}_trex/ fp16 inputIOFormats=fp32:hwc"
os.system(cmd)

engine_name = f"/tmp/{MODEL_NAME}_trex/{MODEL_NAME}.onnx.engine"

plan = trex.EnginePlan(
    f"{engine_name}.graph.json",
    f"{engine_name}.profile.json",
    f"{engine_name}.profile.metadata.json")

plan.summary()

graph = trex.to_dot(plan, trex.layer_type_formatter)
svg_name = trex.render_dot(graph, engine_name, 'svg')
