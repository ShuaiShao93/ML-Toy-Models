import torch

import os
import io
import numpy as np
import time

import torch_mlir
import iree.compiler as ireec
import iree.runtime as ireert

# os.environ["IREE_SAVE_TEMPS"] = "./"

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

# Download an example image from the pytorch website
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
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
model.to(device)

# warmup
with torch.no_grad():
    output = model(input_batch)

start = time.time()
with torch.no_grad():
    output = model(input_batch)
print("torch result", output[0, 0])
print("torch time", time.time() - start)

# IREE
mlir = torch_mlir.compile(
    model,
    input_batch,
    output_type="linalg-on-tensors",
    use_tracing=True)

with open("torch.mlir", "w") as f:
    f.write(str(mlir))

iree_input_type = "tm_tensor"
bytecode_stream = io.BytesIO()
mlir.operation.write_bytecode(bytecode_stream)
flatbuffer = ireec.compile_str(bytecode_stream.getvalue(),
                              target_backends=[device],
                              input_type=iree_input_type,
                              extra_args=[
                                # "--mlir-print-ir-after=iree-flow-form-dispatch-workgroups",
                                # "--mlir-elide-elementsattrs-if-larger=8",
                                "--iree-flow-dump-dispatch-graph",
                                "--iree-flow-dump-dispatch-graph-output-file=foo.dot"])

iree_device = ireert.get_device(device)
config = ireert.Config(device=iree_device)
ctx = ireert.SystemContext(config=config)
vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, flatbuffer)
ctx.add_vm_module(vm_module)
invoker = ctx.modules.module

# warmup
iree_input_batch = ireert.asdevicearray(iree_device, input_batch.cpu().numpy())
result = invoker.forward(iree_input_batch)

start = time.time()
result = invoker.forward(iree_input_batch)
numpy_result = np.asarray(result)
print("iree result", numpy_result[0][0])
print("iree time", time.time() - start)