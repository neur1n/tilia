#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import PIL.Image
import torch
import torchvision
import torchvision.models

import config


# dataset = [
#         # dataset.Dataset("breast-cancer", "classification"),
#         # dataset.Dataset("digits", "classification"),
#         dataset.Dataset("iris", "classification"),
#         ]


if __name__ == "__main__":
    black_box: torch.nn.Module = torch.hub.load(
            "pytorch/vision:v0.20.0",
            "inception_v3",
            weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)  # type: ignore
    black_box.eval()

    image = PIL.Image.open(f"{config.ROOT_DIR}/dataset/maltese_dog.jpg")

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(299),
        torchvision.transforms.CenterCrop(299),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    tensor: torch.Tensor = preprocess(image)  # type: ignore
    batch = tensor.unsqueeze(0)

    if torch.cuda.is_available():
        black_box.to("cuda")
        batch = batch.to("cuda")

    with torch.no_grad():
        output = black_box(batch)

    # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
    print(output[0])

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    # probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(probabilities)
