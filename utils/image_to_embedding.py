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
        transforms.CenterCrop(256),
        # transforms.Resize((256, 256)),  
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

def get_image_to_embedding(frames, model):
    mses, cos_sims = [], []
    curr_latent = tensor_to_flat_latent(frames[0], model)
    for i in range(len(frames) - 1):
        next_latent = tensor_to_flat_latent(frames[i+1], model)
        mse = MSELoss(curr_latent, next_latent)
        cos_sim = cosine_similarity(curr_latent, next_latent)
        mses.append(mse.item())
        cos_sims.append(cos_sim.item())
        curr_latent = next_latent
    
    return np.mean(mses), np.mean(cos_sims)

def get_model():
    model = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    model.eval()
    
    return model