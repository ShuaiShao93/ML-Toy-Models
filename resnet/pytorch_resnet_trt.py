"""
nsys profile --force-overwrite true -w true -t cublas,cudnn,cuda,nvtx,osrt -s cpu -o /tmp/pytorch_resnet_trt python resnet/pytorch_resnet_trt.py
"""

import torch

import time
import os

FP16 = False

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
model.to(device)

if FP16:
    model.half()
    input_batch = input_batch.half()

with torch.no_grad():
    output = model(input_batch)

    start = time.time()
    output = model(input_batch)
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
    os.system("trtexec --onnx=/tmp/ResNet18.onnx --saveEngine=/tmp/ResNet18.trt --fp16")
else:
    os.system("trtexec --onnx=/tmp/ResNet18.onnx --saveEngine=/tmp/ResNet18.trt")