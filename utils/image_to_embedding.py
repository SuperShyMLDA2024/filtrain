import torch
from PIL import Image
from torchvision import transforms
from torch import nn
from diffusers import AutoencoderKL

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