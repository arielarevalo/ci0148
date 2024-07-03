import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def visualize_latent_space(encoder, dataloader, device, use_tsne=False):
    encoder.eval()
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            latents, _ = encoder.forward(images)
            all_latents.append(latents.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_latents = np.concatenate(all_latents, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    if use_tsne:
        reducer = TSNE(n_components=2)
    else:
        reducer = PCA(n_components=2)

    reduced_latents = reducer.fit_transform(all_latents)

    return reduced_latents, all_labels

def log_to_tensorboard(writer, reduced_latents, labels):
    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced_latents[:, 0], reduced_latents[:, 1], c=labels, cmap='viridis', alpha=0.5)
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    writer.add_figure('Latent Space Visualization', fig)
    writer.close()