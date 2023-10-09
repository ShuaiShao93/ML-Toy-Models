"""
nsys profile --force-overwrite true -w true -t cublas,cudnn,cuda,nvtx,osrt -s cpu -o /tmp/pytorch_bert_trt python bert/pytorch_bert_trt.py
"""

import torch

import time
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

BASE_DIR = "/tmp/bert"
ONNX_MODEL_PATH = BASE_DIR + ".onnx"
TRT_MODEL_PATH = BASE_DIR + ".trt"

# tokenize
tokenizer = torch.hub.load(
    'huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
text_1 = "Who was Jim Henson ?"
text_2 = "Jim Henson was a puppeteer"
indexed_tokens = tokenizer.encode(text_1, text_2, add_special_tokens=True)

tokens_tensor = torch.tensor([indexed_tokens])

model = torch.hub.load('huggingface/pytorch-transformers',
                       'model', 'bert-base-cased')
model.eval()

# move the input and model to GPU for speed
device = "cuda"
tokens_tensor = tokens_tensor.to(device)
model.to(device)

with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor)

    start = time.time()
    outputs = model(tokens_tensor)
    output_1 = outputs.last_hidden_state.cpu().numpy()
    output_2 = outputs.pooler_output.cpu().numpy()
    print("torch result", output_1[0, 0, 0])
    print("torch time", (time.time() - start) * 1000)

torch.onnx.export(model,         # model being run
                  # model input (or a tuple for multiple inputs)
                  tokens_tensor,
                  ONNX_MODEL_PATH,       # where to save the model
                  export_params=True,  # store the trained parameter weights inside the model file
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  opset_version=17,  # Require 17 for LayerNorm
                  input_names=['inputs'],   # the model's input names
                  output_names=['outputs', "outputs_2"],
                  dynamic_axes={'inputs': {0: 'batch'}})  # dynamic token length
print("onnx model saved", ONNX_MODEL_PATH)

trt_logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(trt_logger)
EXPLICIT_BATCH = 1 << (int)(
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(EXPLICIT_BATCH)

parser = trt.OnnxParser(network, trt_logger)
with open(ONNX_MODEL_PATH, "rb") as f:
    assert parser.parse(f.read()), "ERROR: Failed to parse the ONNX file."

input_tensor = network.get_input(0)
assert input_tensor.name == "inputs", input_tensor.name

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.DIRECT_IO)
config.profiling_verbosity = trt.ProfilingVerbosity.VERBOSE

opt_profile = builder.create_optimization_profile()
opt_profile.set_shape("inputs", min=(1, 16), opt=(1, 16), max=(8, 16))
config.add_optimization_profile(opt_profile)

serialized_engine = builder.build_serialized_network(network, config)
with open(TRT_MODEL_PATH, 'wb') as f:
    f.write(serialized_engine)

# Load serialized TensorRT model, and run inference.
runtime = trt.Runtime(trt_logger)
with open(TRT_MODEL_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

inspector = engine.create_engine_inspector()
print('trt_engine layer_info:\n{}'.format(
    inspector.get_engine_information(trt.LayerInformationFormat.JSON)
))

context = engine.create_execution_context()

context.set_input_shape("inputs", tokens_tensor.shape)

tokens_tensor = tokens_tensor.cpu().numpy()
d_input = cuda.mem_alloc(tokens_tensor.nbytes)
d_output1 = cuda.mem_alloc(output_1.nbytes)
d_output2 = cuda.mem_alloc(output_2.nbytes)

bindings = [int(d_input), int(d_output1), int(d_output2)]
stream = cuda.Stream()

cuda.memcpy_htod_async(d_input, tokens_tensor, stream)
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
trt_output1, trt_output2 = np.empty_like(output_1), np.empty_like(output_2)
cuda.memcpy_dtoh_async(trt_output1, d_output1, stream)
cuda.memcpy_dtoh_async(trt_output2, d_output2, stream)
stream.synchronize()
print("trt result", trt_output1[0, 0, 0])

assert np.allclose(output_1, trt_output1, atol=1e-2)
assert np.allclose(output_2, trt_output2, atol=1e-2)
