import cv2
import numpy as np

def get_optical_flow(frames):
    avg_velocities = []
    prv = frames[0].squeeze(0).cpu().numpy().transpose((1, 2, 0))
    prv_gray = cv2.cvtColor(prv, cv2.COLOR_BGR2GRAY)

    for i in range(len(frames)-1):
        nxt = frames[i+1].squeeze(0).cpu().numpy().transpose((1, 2, 0))
        nxt_gray = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prv_gray, nxt_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Calculate the magnitude and angle of the 2D vectors
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Calculate the average magnitude of the vectors, which corresponds to the average velocity
        average_velocity = np.mean(magnitude)
        avg_velocities.append(average_velocity)

        # Update the previous frame
        prv = nxt

    return np.mean(avg_velocities)