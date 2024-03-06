import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def tensor_to_flat_latent(tensor, model):
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

def get_image_to_embedding(frames, model, device):
    final_mse = 0
    final_cos_sim = 0

    curr_latent = tensor_to_flat_latent(frames[0].to(device), model).cpu().numpy()
    for i in range(len(frames) - 1):
        next_latent = tensor_to_flat_latent(frames[i+1].to(device), model).cpu().numpy()
        mse = MSELoss(curr_latent, next_latent)
        cos_sim = cosine_similarity(curr_latent, next_latent)
        final_mse += mse
        final_cos_sim += cos_sim
        curr_latent = next_latent
    
    final_mse = final_mse / (len(frames)-1)
    final_cos_sim = final_cos_sim / (len(frames)-1)
    return final_mse, final_cos_sim