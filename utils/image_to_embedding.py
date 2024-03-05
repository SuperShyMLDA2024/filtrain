import torch
from PIL import Image
from torchvision import transforms
from torch import nn
from diffusers import AutoencoderKL

import os
import json
import cv2
import numpy as np
import glob
from dataset_class_batch import VideoDataset 
from split_clip import split_clip

def image_to_tensor(image_path):

    # Load the image using Pillow
    img = Image.open(image_path)

    # Convert the image to RGB format (if necessary)
    img = img.convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),  
    ])

    return transform(img).unsqueeze(0)

def tensor_to_flat_latent(tensor, model):
    # Dont forget to pass the model below
    # model = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    # model.eval()
    y = model.encoder(tensor)
    return y.view(1, -1)

def MSELoss(x1, x2):
    loss = nn.MSELoss()
    return loss(x1, x2)

def cosine_similarity(x1, x2):
    cosine_score = nn.CosineSimilarity()
    return cosine_score(x1, x2)


if __name__ == '__main__':
    # Load the model, put it in the GPU
    model = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    model.eval()

    tensors = []
    latents = []

    for image_path in glob.glob('./inference_testing/*'):
        tensor = image_to_tensor(image_path)
        latent = tensor_to_flat_latent(tensor, model)
        tensors.append(tensor)
        latents.append(latent)
    
    for i in range(len(latents)):
        for j in range(i+1, len(latents)):
            mse = MSELoss(latents[i], latents[j])
            cos_sim = cosine_similarity(latents[i], latents[j])
            print(f"MSE between photo {i+1} and photo {j+1}: {mse}")
            print(f"Cosine similarity between photo {i+1} and photo {j+1}: {cos_sim}\n")
        