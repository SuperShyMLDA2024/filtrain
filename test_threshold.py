from utils.image_to_embedding import get_image_to_embedding
from utils.static_check import get_static_difference
from utils.optical_flow_check import get_optical_flow
from diffusers import AutoencoderKL
import torch

import cv2
import numpy as np
from PIL import Image
import os
from torchvision import transforms

def image_transform(image):
    transform = transforms.Compose([
        transforms.Resize((320, 240)),
        transforms.CenterCrop(240),
        transforms.ToTensor(),  
    ])
    return transform(image).unsqueeze(0)

def load_image(folder_path):
    # Sort the frames
    frames = sorted(os.listdir(folder_path))
    conv_frames = []
    
    # Convert the frames to tensor
    for frame in frames:
        image = Image.open(os.path.join(folder_path, frame)).convert('RGB')
        image = image_transform(image)
        conv_frames.append(image)
    return conv_frames

def get_model():
    model = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    return model

def get_inference(folder_path, model, device):
    frames = load_image(folder_path)

    # Getting static difference
    rgb_diff = get_static_difference(frames)

    # getting optical flow
    avg_velocity = get_optical_flow(frames)
    
    # Getting image context similarity
    frame_mse, frame_cos_sim = get_image_to_embedding(frames, model, device)

    print("Folder Path: ", folder_path)
    print("RGB Difference: ", rgb_diff)
    print("Average Velocity: ", avg_velocity)
    print("MSE: ", frame_mse)
    print("Cosine Similarity: ", frame_cos_sim)
    return rgb_diff, avg_velocity, frame_mse, frame_cos_sim

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model().to(device)

    img1 = "frames_output/1gul68uPqQk/1gul68uPqQk.4_3"
    # Car with moving background (Smooth)
    # RGB Difference:  160.80394
    # Average Velocity:  2.7397819
    # MSE:  5.284577873018053
    # Cosine Similarity:  0.8974157108200921
    img2 = "frames_output/1NRXqc74kQM/1NRXqc74kQM.4_1"
    # Dude not moving much (AFK)
    # RGB Difference:  85.367805
    # Average Velocity:  0.8525288
    # MSE:  4.593294726176695
    # Cosine Similarity:  0.9460431228984486
    img3 = "frames_output/3DlCGwJodqg/3DlCGwJodqg.1_0"
    # Huge camera movement (A bit shaky)
    # RGB Difference:  7062.2065
    # Average Velocity:  7.7320466
    # MSE:  32.389395627108485
    # Cosine Similarity:  0.6090081957253543
    img4 = "frames_output/-bmS0RumV9U/-bmS0RumV9U.10_1"
    # Too many sudden transitions
    # RGB Difference:  4791.261
    # Average Velocity:  4.6297765
    # MSE:  10.938308153396997
    # Cosine Similarity:  0.8629428820732312


    get_inference(img1, model, device)
    get_inference(img2, model, device)
    get_inference(img3, model, device)
    get_inference(img4, model, device)