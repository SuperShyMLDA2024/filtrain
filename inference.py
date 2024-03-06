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

from diffusers import AutoencoderKL
from utils.dataset_class_batch import VideoDataset
from utils.image_to_embedding import get_image_to_embedding
from utils.static_check import get_static_difference
from utils.optical_flow_check import get_optical_flow

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

def get_model():
    model = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    return model

def run(dataset, model):
    res = {}
    for data in dataset:
        starttime = time.time()

        scene_id = data['scene_id']
        print(f'Processing scene id: {scene_id}')
        frames_path = data['frames_path']
        frames = load_image(frames_path)
        
        # Assert clips with less than 2 frames
        if(len(frames) < 2):
            assert(f'Scene id {scene_id} has less than 2 frames')
        
        # Getting static difference
        static_diff = get_static_difference(frames)
        
        # Getting image context similarity
        frame_mse, frame_cos_sim = get_image_to_embedding(frames, model)

        # getting optical flow
        avg_velocity = get_optical_flow(frames)
        
        # Storing the results
        if scene_id in res:
            res[scene_id]['static_diff'].append(static_diff)
            res[scene_id]['mse'].append(frame_mse)
            res[scene_id]['cos_sim'].append(frame_cos_sim)
            res[scene_id]['avg_velocity'].append(avg_velocity)
        else:
            res[scene_id] = {'static_diff': [static_diff], 
                             'mse': [frame_mse], 
                             'cos_sim': [frame_cos_sim], 
                             'avg_velocity': [avg_velocity]}
        
        print(f'Processing time for scene id {scene_id}: {time.time() - starttime}')
    
    
    # Taking the average result for each clip
    for scene_id in res:
        res[scene_id]['static_diff'] = float(np.mean(res[scene_id]['static_diff']))
        res[scene_id]['mse'] = float(np.mean(res[scene_id]['mse']))
        res[scene_id]['cos_sim'] = float(np.mean(res[scene_id]['cos_sim']))
        res[scene_id]['avg_velocity'] = float(np.mean(res[scene_id]['avg_velocity']))
    
    return res


# returning the inference result in the form of
# {'clip_id': {'static_diff': static_diff, 'mse': mse, 'cos_sim': cos_sim, 'avg_velocity': avg_velocity}}

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'using device: {device}')

    metafile_path = './metafiles/hdvg_0.json'
    with open(metafile_path, 'r') as f:
        data = json.load(f)

    model = get_model()
    model.to(device)
    
    inference_output_dir = 'frame_score_results'
    if not os.path.exists(inference_output_dir):
        os.makedirs(inference_output_dir)

    N_VIDEOS_PER_BATCH = 5
    for i in range(0, 10, N_VIDEOS_PER_BATCH):
        j = i + N_VIDEOS_PER_BATCH - 1
        print(f'Processing Video {i}-{j}')
        starttime = time.time()
        dataset = VideoDataset(data, i, j)
        
        res = run(dataset, model)
        print(f'Total time: {time.time() - starttime}')
        print(res)
        
        with open(os.path.join(inference_output_dir, f'inference_result_{i}-{j}.json'), 'w') as f:
            json.dump(res, f)