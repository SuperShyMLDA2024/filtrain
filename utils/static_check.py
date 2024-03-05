import os
import json
import cv2
import numpy as np
import glob
from dataset_class_batch import VideoDataset 
from split_clip import split_clip


temp_path = './temp.json'
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

if __name__ == '__main__':
    with open("metafiles/hdvg_0.json", 'r') as f:
        data = json.load(f)
    
    dataset = VideoDataset(data, 0, 0)
    print(len(dataset))
    for data in dataset:
        clip_id = data['clip_name']
        video_id = data['video_id']
        print(video_id, clip_id)
        print("------------------------------------------------------------")
    print("Done!")