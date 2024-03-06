import os
import cv2
import numpy as np
import torch

def get_static_difference(frames):
    diff = []
    for i in range(len(frames)-1):
        frame1 = frames[i] * 255
        frame2 = frames[i+1] * 255
        diff.append(torch.mean(torch.square(frame1 - frame2)))
    return np.max(diff)