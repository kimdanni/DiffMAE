import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToTensor, Normalize, Compose
from PIL import Image
from diffusers import DDPMScheduler

# Assuming you have a method to load a test image
def load_test_image(path):
    image = Image.open(path)
    transformations = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transformations(image).unsqueeze(0)

# Load the test image
image = load_test_image('/home/danni/workspace/CrossMAE/cat1.jpg').to("cuda:0")
diffusion_timestpes = 1000  # Number of Diffusion Steps during training. Note, this is much smaller than ordinary (usually ~1000)
beta_start = 1e-4 # Default from paper
beta_end = 0.02  # NOTE: This is different to accomodate the much shorter DIFFUSION_TIMESTEPS (Usually ~1000). For 1000 diffusion timesteps, use 0.02.
clip_sample = False  # Used for better sample generation
noise_scheduler = DDPMScheduler(num_train_timesteps=diffusion_timestpes, beta_schedule="linear", beta_start=beta_start, beta_end=beta_end, clip_sample=clip_sample)
t = torch.arange(0, noise_scheduler.num_train_timesteps, dtype=torch.long, device=image.device).unsqueeze(1).expand(diffusion_timestpes, -1)
noise = torch.rand_like(image)
noise_x = noise_scheduler.add_noise(image, noise, t)

import matplotlib.pyplot as plt
import numpy as np
import math

# Select every 10th image
selected_images = noise_x[::4]

# Move the tensors to CPU and convert them to numpy arrays
selected_images = selected_images.permute(0, 2, 3, 1).cpu().detach().numpy()

# Determine the size of the grid
grid_size = int(math.sqrt(selected_images.shape[0]))

# Create a figure to display the images
fig, axs = plt.subplots(grid_size, grid_size, figsize=(20, 20))

for i, img in enumerate(selected_images):
    # Normalize the image to [0, 1] range
    if i >= grid_size * grid_size:
        continue
    img = (img - img.min()) / (img.max() - img.min())
    axs[i // grid_size, i % grid_size].imshow(img)
    axs[i // grid_size, i % grid_size].axis('off')

plt.show()
