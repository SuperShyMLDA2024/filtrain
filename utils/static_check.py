import os
import json
import cv2
import numpy as np
import glob
from dataset_class_batch import VideoDataset 
from split_clip import split_clip


temp_path = './temp.json'
enter = "-" * 20
def convert_image(folder_path):
    frames = sorted(os.listdir(folder_path))
    conv_frames = []
    
    for frame in frames:
        image = cv2.imread(os.path.join(folder_path, frame))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        conv_frames.append(image)
        
    return np.array(conv_frames)

def get_static_difference(frames):
    diff = []
    for i in range(len(frames)-1):
        diff.append(np.mean((frames[i] - frames[i+1])**2))
    return np.mean(diff)


def run(metafile_path, min_idx, max_idx):
    with open(metafile_path, 'r') as f:
        data = json.load(f)
    
    dataset = VideoDataset(data, min_idx, max_idx)
    static_diffs = {}
    for data in dataset:
        clip_id = data['clip_id']
        frames_path = data['frames_path']
        frames = convert_image(frames_path)
        static_diff = get_static_difference(frames)
        if frames.shape[0] > 1:
            if clip_id in static_diffs:
                static_diffs[clip_id].append(static_diff)
            else:
                static_diffs[clip_id] = [static_diff]
    
    for clip_id in static_diffs:
        if len(static_diffs[clip_id]):
            static_diffs[clip_id] = np.mean(static_diffs[clip_id])
        else:
            static_diffs[clip_id] = np.inf
    
    return static_diffs

if __name__ == '__main__':
    res = run('./metafiles/hdvg_batch_0-1.json', 0, 4)
    print(res)
    print("Done!")