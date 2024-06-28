import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split

class PlantVillage(Dataset):
    def __init__(self, tf_dataset):
        self.tf_dataset = tf_dataset
        self.data = list(tf_dataset.as_numpy_iterator())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]['image']
        label = self.data[idx]['label']
        
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return data, label

def get_splits(dataset, batch_size=32):
    """Gets the data splits for training, validation, and testing.

    Args:
        dataset (Dataset): Dataset to produce splits from
        batch_size (int, optional): Batch size for the data loaders. Defaults to 32.

    Returns:
        tuple: Tuple containing training, validation, and test data loaders.
    """
    # Get the labels
    labels = np.array([label for _, label in dataset])

    train_indices, temp_indices = train_test_split(np.arange(len(dataset)), test_size=0.2, stratify=labels)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, stratify=labels[temp_indices])

    # Create Samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create DataLoaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    return train_loader, val_loader, test_loader