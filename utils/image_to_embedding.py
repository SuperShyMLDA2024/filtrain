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
import numpy as np

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
    with torch.inference_mode():
        y = model.encoder(tensor)
    return y.flatten()

def MSELoss(x1, x2):
    loss = np.mean((x1 - x2) ** 2)
    return loss

def cosine_similarity(x1, x2):
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    cosine_score = np.dot(x1, x2) / (norm_x1 * norm_x2)
    return cosine_score

def get_image_to_embedding(frames, model):
    mses, cos_sims = [], []
    final_mse = 0
    final_cos_sim = 0

    curr_latent = tensor_to_flat_latent(frames[0], model).cpu().numpy()
    for i in range(len(frames) - 1):
        next_latent = tensor_to_flat_latent(frames[i+1], model).cpu().numpy()
        mse = MSELoss(curr_latent, next_latent)
        cos_sim = cosine_similarity(curr_latent, next_latent)
        final_mse += mse
        final_cos_sim += cos_sim
        curr_latent = next_latent
    
    final_mse = final_mse / (len(frames)-1)
    final_cos_sim = final_cos_sim / (len(frames)-1)
    return final_mse, final_cos_sim

def get_model():
    model = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    
    return model