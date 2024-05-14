import matplotlib.pyplot as plt

def plot_sample(dataset, idx):
    """
    Plot a sample image and its mask from the dataset.

    Args:
        dataset (Dataset): The custom BCSSDataset instance.
        idx (int): Index of the sample to be plotted.
    """
    image, mask = dataset[idx]
    image_np = image.permute(1, 2, 0).numpy()  # Convert from PyTorch tensor to numpy array
    mask_np = mask.squeeze().numpy()  # Remove channel dimension and convert to numpy array

    plt.figure(figsize=(12, 6))

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title('Original Image')
    plt.axis('off')

    # Plot image with mask overlay
    plt.subplot(1, 2, 2)
    plt.imshow(image_np)
    plt.imshow(mask_np, alpha=0.8)  # Alpha controls the transparency
    plt.title('Image with Mask Overlay')
    plt.axis('off')

    plt.show()

# Plot some samples from the dataset
[plot_sample(val_dataset, np.random.randint(0, len(val_dataset) - 1)) for i in range(10)]
