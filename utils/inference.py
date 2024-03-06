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
from image_to_embedding import get_model, get_image_to_embedding
from static_check import get_static_difference

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

def convert_image(folder_path):
    # Sort the frames
    frames = sorted(os.listdir(folder_path))
    conv_frames = []
    
    # Convert the frames to tensor
    for frame in frames:
        image = image_to_tensor(os.path.join(folder_path, frame))
        image = image.to(device) # Move tensor to GPU
        conv_frames.append(image)
    return conv_frames


def get_dataset(metafile_path, min_idx, max_idx):
    with open(metafile_path, 'r') as f:
        data = json.load(f)
    
    dataset = VideoDataset(data, min_idx, max_idx)
    return dataset

def run(dataset, model):
    
    res = {}
    for data in dataset:
        starttime = time.time()

        clip_id = data['clip_id']
        print(f'Processing clip id: {clip_id}')
        frames_path = data['frames_path']
        frames = convert_image(frames_path)
        
        # Skipping clips with less than 2 frames
        if(len(frames) < 2):
            continue
        
        # Getting static difference
        static_diff = get_static_difference(frames)
        
        # Getting image context similarity
        frame_mse, frame_cos_sim = get_image_to_embedding(frames, model)
        
        # Storing the results
        if clip_id in res:
            res[clip_id]['static_diff'].append(static_diff)
            res[clip_id]['mse'].append(frame_mse)
            res[clip_id]['cos_sim'].append(frame_cos_sim)
        else:
            res[clip_id] = {'static_diff': [static_diff], 'mse': [frame_mse], 'cos_sim': [frame_cos_sim]}
        
        print(f'Processing time for clip id {clip_id}: {time.time() - starttime}')
    
    
    # Taking the average result for each clip
    for clip_id in res:
        res[clip_id]['static_diff'] = float(np.mean(res[clip_id]['static_diff']))
        res[clip_id]['mse'] = float(np.mean(res[clip_id]['mse']))
        res[clip_id]['cos_sim'] = float(np.mean(res[clip_id]['cos_sim']))
    print(res)
    return res


# returning the inference result in the form of
# {'clip_id': {'static_diff': static_diff, 'mse': mse, 'cos_sim': cos_sim}}

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'using device: {device}')

    starttime = time.time()
    dataset = get_dataset('./metafiles/hdvg_0.json', 0, 1)
    model = get_model()
    model.to(device)
    model.eval()
    
    res = run(dataset, model)
    print(f'Total time: {time.time() - starttime}')
    
    # Create the directory if it doesn't exist
    os.makedirs('./frame_score_results', exist_ok=True)
    
    # Dumping the result as JSON
    with open('./frame_score_results/inference_result.json', 'w') as f:
        json.dump(res, f)