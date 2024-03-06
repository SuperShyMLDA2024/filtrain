import cv2
import numpy as np
from PIL import Image
import os
from torchvision import transforms

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

def get_optical_flow(frames):
    avg_velocities = []
    prv = frames[0].squeeze(0).numpy().transpose((1, 2, 0)) * 255
    prv_gray = cv2.cvtColor(prv, cv2.COLOR_BGR2GRAY)

    for i in range(len(frames)-1):
        nxt = frames[i+1].squeeze(0).numpy().transpose((1, 2, 0)) * 255
        nxt_gray = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prv_gray, nxt_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Calculate the magnitude and angle of the 2D vectors
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Calculate the average magnitude of the vectors, which corresponds to the average velocity
        average_velocity = np.mean(magnitude)
        avg_velocities.append(average_velocity)

        # Update the previous frame
        prv = nxt

    return np.max(avg_velocities), np.mean(avg_velocities)

if __name__ == '__main__':
    # Test the function
    
    frames_1 = load_image("frames_output/1gul68uPqQk/1gul68uPqQk.4_3")
    print(get_optical_flow(frames_1))

    frames_2 = load_image("frames_output/1NRXqc74kQM/1NRXqc74kQM.4_1")
    print(get_optical_flow(frames_2))

    frames_3 = load_image("frames_output/3DlCGwJodqg/3DlCGwJodqg.1_0")
    print(get_optical_flow(frames_3))

    frames_4 = load_image("frames_output/-bmS0RumV9U/-bmS0RumV9U.10_1")
    print(get_optical_flow(frames_4))