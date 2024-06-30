from torchvision.transforms import Lambda
import matplotlib.pyplot as plt
import numpy as np
import random
def visualize_noise_image(noise_x):
    unnormalize = Lambda(lambda t: (t + 1) / 2)
    noise_x_unnorm = unnormalize(noise_x)
    # Convert the tensor to a numpy array and normalize it to [0, 255]
    images_np = (noise_x_unnorm.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(5, 10, figsize=(20, 10))

    # Plot the first 50 images
    for i, ax in enumerate(axes.flat):
        if i < 50:
            ax.imshow(images_np[i])
            ax.axis('off')  # Hide the axes
    image_name = random.randint(0, 1000)
    # Save the image
    plt.savefig(f'noise_batch{image_name}.png')