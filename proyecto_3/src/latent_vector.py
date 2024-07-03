import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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
        reducer = TSNE(n_components=3)
    else:
        reducer = PCA(n_components=3)

    reduced_latents = reducer.fit_transform(all_latents)

    return reduced_latents, all_labels


def create_metadata_list(labels, idx_to_class):
    metadata = [idx_to_class[label] for label in labels]
    return metadata


def log_embeddings(writer, embeddings, metadata):
    writer.add_embedding(mat=embeddings, metadata=metadata)
    writer.close()
