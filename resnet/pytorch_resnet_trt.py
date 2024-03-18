"""
nsys profile --force-overwrite true -w true -t cublas,cudnn,cuda,nvtx,osrt -s cpu -o /tmp/pytorch_resnet_trt python3 resnet/pytorch_resnet_trt.py
"""

import torch

import time
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

FP16 = True
TRT_MODEL_PATH = "/tmp/ResNet18.trt"

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

# Download an example image from the pytorch website
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "/tmp/dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed
device = "cuda"
input_batch = input_batch.to(device)
model.eval().to(device)

with torch.no_grad():
    output = model(input_batch)

    start = time.time()
    output = model(input_batch)
    print("torch result shape", output)
    print("torch result", output[0, 0])
    print("torch time", time.time() - start)

torch.onnx.export(model,         # model being run 
    input_batch,       # model input (or a tuple for multiple inputs) 
    "/tmp/ResNet18.onnx",       # where to save the model  
    export_params=True,  # store the trained parameter weights inside the model file 
    do_constant_folding=True,  # whether to execute constant folding for optimization 
    input_names = ['modelInput'],   # the model's input names 
    output_names = ['modelOutput'])

if FP16:
    os.system(f"trtexec --onnx=/tmp/ResNet18.onnx --saveEngine={TRT_MODEL_PATH} --fp16")
else:
    os.system(f"trtexec --onnx=/tmp/ResNet18.onnx --saveEngine={TRT_MODEL_PATH}")

# Load serialized TensorRT model, and run inference.
trt_logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(trt_logger)
with open(TRT_MODEL_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

# inspector = engine.create_engine_inspector()
# print('trt_engine layer_info:\n{}'.format(
#     inspector.get_engine_information(trt.LayerInformationFormat.JSON)
#     ))

context = engine.create_execution_context()

input_batch = input_batch.cpu().numpy()
output = output.cpu().numpy()

d_input = cuda.mem_alloc(1 * input_batch.nbytes)
trt_output = np.empty_like(output)
d_output = cuda.mem_alloc(1 * trt_output.nbytes)

bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

cuda.memcpy_htod_async(d_input, input_batch, stream)
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
cuda.memcpy_dtoh_async(trt_output, d_output, stream)
stream.synchronize()
print("trt result", trt_output[0, 0])

assert np.allclose(output, trt_output, atol=5e-2)