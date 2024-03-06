import torch
import numpy as np
import time


# Example ground truth and predicted values
ground_truth = torch.tensor([1, 2, 3]).to('cuda')
predictions = torch.tensor([2, 2, 4]).to('cuda')

time1 = time.time()

# PyTorch MSE loss calculation
mse_loss_torch = torch.nn.MSELoss()
loss_torch = mse_loss_torch(predictions.float(), ground_truth.float())
print("PyTorch MSE Loss:", loss_torch.item())

time2 = time.time()

# NumPy MSE loss calculation
mse_loss_numpy = np.mean((predictions.cpu().numpy() - ground_truth.cpu().numpy()) ** 2)
print("NumPy MSE Loss:", mse_loss_numpy)

time3 = time.time()

print("PyTorch Time:", time2 - time1)
print("NumPy Time:", time3 - time2)