from utils.filter_scenes import filter_scenes
from utils.optical_flow_check import get_optical_flow
from utils.static_check import get_static_difference
from utils.image_to_embedding import get_image_to_embedding
from utils.dataset_class_batch import VideoDataset
from diffusers import AutoencoderKL
import json
import os
from PIL import Image
import torch
from torchvision import transforms
import time
import pickle
import warnings
warnings.filterwarnings("ignore")
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
    for data in dataset:
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
            'info': data
        }

        print(f'Processing time for scene id {scene_id}: {time.time() - starttime}')

    return res


# returning the inference result in the form of
# {'clip_id': {'static_diff': static_diff, ...}}

N_VIDEOS_PER_BATCH = 4
N_TOTAL_VIDEOS = 18_750
N_TOTAL_CLIPS = 1_500_000
TOTAL_CLIPS_TAKEN = 10_000
CLIPS_IDX_START = 100
LENGTH = 4
CLIPS_IDX_END = CLIPS_IDX_START + LENGTH - 1
CLIPS_TAKEN_PER_BATCH = max(1, int(N_VIDEOS_PER_BATCH / N_TOTAL_VIDEOS * TOTAL_CLIPS_TAKEN))
metafile_path = './metafiles/hdvg_0.json'
filename = 'xgboost_model.pth'

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'using device: {device}')
    print(f'CLIPS_TAKEN_PER_BATCH: {CLIPS_TAKEN_PER_BATCH}')

    with open(metafile_path, 'r') as f:
        data = json.load(f)

    model = get_model()
    model.to(device)

    inference_output_dir = 'frame_score_results'
    if not os.path.exists(inference_output_dir):
        os.makedirs(inference_output_dir)

    classifier_model = pickle.load(open(filename, 'rb'))

    for i in range(CLIPS_IDX_START, CLIPS_IDX_END+1, N_VIDEOS_PER_BATCH):
        j = i + N_VIDEOS_PER_BATCH - 1
        print(f'Processing Video {i}-{j}')
        starttime = time.time()

        dataset = VideoDataset(data, i, j)
        res = get_metrics(dataset, model, device)

        # save res to json for DEBUGGING
        # with open(os.path.join(inference_output_dir, f'inference_result_{i}-{j}.json'), 'w') as f:
        #     json.dump(res, f)

        filtered_scenes = filter_scenes(res, CLIPS_TAKEN_PER_BATCH, classifier_model)
        print("No. Scenes Taken:", len(filtered_scenes))
        print(f'Total time: {time.time() - starttime}')

        json_info = {
            'length': len(filtered_scenes),
            'filtered_scenes': filtered_scenes
        }
        with open(os.path.join(inference_output_dir, f'filtered_scenes_{i}-{j}.json'), 'w') as f:
            json.dump(json_info, f)
