import json
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import warnings
import time
warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from dataset_class_batch import VideoDataset
from image_to_embedding import get_image_to_embedding
from static_check import get_static_difference
from diffusers import AutoencoderKL

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
        image = image_transform(image).to(device)
        conv_frames.append(image)
    return conv_frames


def get_dataset(metafile_path, min_idx, max_idx):
    with open(metafile_path, 'r') as f:
        data = json.load(f)
    
    dataset = VideoDataset(data, min_idx, max_idx)
    return dataset

def get_model():
    model = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    return model

def run(dataset, model):
    res = {}
    for data in dataset:
        starttime = time.time()

        scene_id = data['scene_id']
        print(f'Processing clip id: {scene_id}')
        frames_path = data['frames_path']
        frames = load_image(frames_path)
        
        # Assert clips with less than 2 frames
        if(len(frames) < 2):
            assert(f'Scene id {scene_id} has less than 2 frames')
        
        # Getting static difference
        static_diff = get_static_difference(frames)
        
        # Getting image context similarity
        frame_mse, frame_cos_sim = get_image_to_embedding(frames, model)
        
        # Storing the results
        if scene_id in res:
            res[scene_id]['static_diff'].append(static_diff)
            res[scene_id]['mse'].append(frame_mse)
            res[scene_id]['cos_sim'].append(frame_cos_sim)
        else:
            res[scene_id] = {'static_diff': [static_diff], 'mse': [frame_mse], 'cos_sim': [frame_cos_sim]}
        
        print(f'Processing time for scene id {scene_id}: {time.time() - starttime}')
    
    
    # Taking the average result for each clip
    for clip_id in res:
        res[clip_id]['static_diff'] = np.mean(res[clip_id]['static_diff'])
        res[clip_id]['mse'] = np.mean(res[clip_id]['mse'])
        res[clip_id]['cos_sim'] = np.mean(res[clip_id]['cos_sim'])
    
    return res


# returning the inference result in the form of
# {'clip_id': {'static_diff': static_diff, 'mse': mse, 'cos_sim': cos_sim}}

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'using device: {device}')

    starttime = time.time()
    dataset = get_dataset('./metafiles/hdvg_0.json', 0, 9)
    model = get_model()
    model.to(device)
    
    res = run(dataset, model)
    print(f'Total time: {time.time() - starttime}')
    print(res)