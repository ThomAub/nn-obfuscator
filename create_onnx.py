"""Create some onnx files to use in the project."""
import pathlib

import torch
import torch.nn as nn
from loguru import logger
from torchvision import models

output_path = pathlib.Path("tests/data/saved_models")
output_pytorch_path = output_path / "pytorch"
output_onnx_path = output_path / "onnx"

output_pytorch_path.mkdir(parents=True, exist_ok=True)
output_onnx_path.mkdir(parents=True, exist_ok=True)

small_architecture_list = [
    "default",
    "simple_conv",
    "simple_conv_relu",
    "simple_linear",
    "simple_linear_relu",
    "vgg11",
    "vgg11_bn",
]
large_architecture_list = small_architecture_list + [
    "vgg16",
    "vgg16_bn",
    "resnet18",
    "resnet50",
    "resnet101",
    "inception_v3",
    "densenet121",
    "mobilenet_v2",
]

# Removing "mobilenet_v3_small", "mobilenet_v3_large" due to:  hardsigmoid to ONNX opset version 12 is not supported
# https://github.com/pytorch/vision/issues/3463
if __name__ == "__main__":

    for architecture in large_architecture_list:
        if "default" in architecture:
            dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
            model = torch.nn.Sequential(nn.Conv2d(3, 20, 5), nn.ReLU(), nn.Conv2d(20, 64, 5), nn.ReLU())
        elif "simple_conv" in architecture:
            dummy_input = torch.randn(1, 3, 6, 6, device="cpu")
            if "relu" in architecture:
                model = torch.nn.Sequential(nn.Conv2d(3, 6, 5), nn.ReLU())
            else:
                model = torch.nn.Sequential(nn.Conv2d(3, 6, 5))
        elif "simple_linear" in architecture:
            dummy_input = torch.randn(1, 3, device="cpu")
            if "relu" in architecture:
                model = torch.nn.Sequential(nn.Linear(3, 6), nn.ReLU())
            else:
                model = torch.nn.Sequential(nn.Linear(3, 6))
        else:
            dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
            model = models.__dict__[architecture]()

        logger.debug(
            f"Using {architecture}\nSaving pytorch model to: {output_pytorch_path.as_posix()}/{architecture}_torch.pt\nSaving ONNX model to: {output_onnx_path.as_posix()}/{architecture}.onnx"
        )
        model = model.eval()
        torch.save(model, f"{output_pytorch_path.as_posix()}/{architecture}_torch.pt")
        torch.onnx.export(model, dummy_input, f"{output_onnx_path.as_posix()}/{architecture}.onnx", opset_version=11)
