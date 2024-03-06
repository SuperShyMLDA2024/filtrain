import os
import cv2
import numpy as np
import torch

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
        frame1 = frames[i]
        frame2 = frames[i+1]
        diff.append(torch.mean(torch.square(frame1 - frame2)))
    return np.mean(diff)