from utils.filter_scenes import filter_scenes
from utils.optical_flow_check import get_optical_flow
from utils.static_check import get_static_difference
from utils.image_to_embedding import get_image_to_embedding
from utils.dataset_class_batch import VideoDataset
from utils.gemini_recaptioning import GeminiRecaptioning

from diffusers import AutoencoderKL
import json
import os
from PIL import Image
import torch
from torchvision import transforms
import time
import pickle
import yaml

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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


def get_metrics(dataset, model, device):
    res = {}
    for idx, data in enumerate(dataset):
        starttime = time.time()

        scene_id = data['scene_id']
        frames_path = data['frames_path']
        frames = load_image(frames_path)

        # Getting static difference
        max_rgb_diff, mean_rgb_diff = get_static_difference(frames)

        # getting optical flow
        max_velocity, mean_velocity = get_optical_flow(frames)

        # Getting image context similarity
        frame_mse, frame_cos_sim = get_image_to_embedding(frames, model, device)

        # No. frames
        no_frames = len(frames)

        # Storing the results
        res[scene_id] = {
            'max_rgb_diff': float(max_rgb_diff),
            'mean_rgb_diff': float(mean_rgb_diff),
            'max_velocity': float(max_velocity),
            'mean_velocity': float(mean_velocity),
            'mse': float(frame_mse),
            'cos_sim': float(frame_cos_sim),
            'no_frames': float(no_frames),
            'idx': idx,
            'clip_id': data['clip_id'],
        }

        print(f'Processing time for scene id {scene_id}: {time.time() - starttime}')

    return res


# returning the inference result in the form of
# {'clip_id': {'static_diff': static_diff, ...}}

N_TOTAL_VIDEOS = 18_750
N_TOTAL_CLIPS = 1_500_000
TOTAL_CLIPS_TAKEN = 10_000
metafile_path = './metafiles/hdvg_0.json'
classifier_filename = 'xgboost_model.pkl'
api_key = os.getenv("GEMINI_API_KEY")

if __name__ == '__main__':
    # Load the settings from the YAML file
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'using device: {device}')

    # Parse the arguments
    n_videos_per_batch = config["n_videos_per_batch"]
    clip_idx_start = config["clip_idx_start"]
    clip_idx_end = config["clip_idx_end"]

    CLIPS_TAKEN_PER_BATCH = max(1, int(n_videos_per_batch / N_TOTAL_VIDEOS * TOTAL_CLIPS_TAKEN))
    print(f'CLIPS_TAKEN_PER_BATCH: {CLIPS_TAKEN_PER_BATCH}')

    with open(metafile_path, 'r') as f:
        data = json.load(f)

    model = get_model()
    model.to(device)

    inference_output_dir = 'frame_score_results'
    if not os.path.exists(inference_output_dir):
        os.makedirs(inference_output_dir)

    classifier_model = pickle.load(open(classifier_filename, 'rb'))
    
    # Create an instance of the GeminiRecaptioning class
    gemini_recaptioning = GeminiRecaptioning(api_key, data)

    for i in range(clip_idx_start, clip_idx_end+1, n_videos_per_batch):
        j = i + n_videos_per_batch - 1
        print(f'Processing Video {i}-{j}')
        starttime = time.time()

        dataset = VideoDataset(data, i, j)
        res = get_metrics(dataset, model, device)

        # save the result
        with open(os.path.join(inference_output_dir, f'inference_result_{i}-{j}.json'), 'w') as f:
            json.dump(res, f)

        filtered_scenes = filter_scenes(res, CLIPS_TAKEN_PER_BATCH, classifier_model)
        print("No. Scenes Taken:", len(filtered_scenes))

        filtered_scenes = [dataset[idx] for idx in filtered_scenes]
        for scene in filtered_scenes:
            frames_path = scene['frames_path']
            assert(os.path.exists(frames_path))
            print(f"Recaptioning for {frames_path}")
            
            # Initialize the recaption variable
            recaption = ""

            # Try the operation three times max
            for _ in range(3):
                try:
                    recaption = gemini_recaptioning.run(frames_path).strip()
                    # If the operation is successful, break out of the loop
                    break
                except:
                    # If the operation is not successful, continue to the next iteration
                    continue

            scene['recaption'] = recaption

        json_info = {
            'length': len(filtered_scenes),
            'filtered_scenes': filtered_scenes
        }

        print(f'Total time: {(time.time() - starttime):.2f}')

        json_filename = f'filtered_scenes_{i}-{j}.json'
        with open(os.path.join(inference_output_dir, json_filename), 'w') as f:
            json.dump(json_info, f)
        
        print(f"Saved to json: {json_filename}")
