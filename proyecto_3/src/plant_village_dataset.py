import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm


class PlantVillageDataset(ImageFolder):
    initial_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    def __init__(self, root):
        print('Loading Plant Village')
        super(PlantVillageDataset, self).__init__(root, transform=self.initial_transform)

        print(' - Normalizing dataset')
        dataloader = DataLoader(self, batch_size=128, shuffle=False, num_workers=4)

        # Initialize variables to store the sum of means and stds
        mean = torch.zeros(3)  # for RGB channels
        std = torch.zeros(3)
        nb_samples = 0

        # Iterate over the dataset
        for data, _ in tqdm(dataloader, desc=' - Calculating mean and standard deviation', unit='batch'):
            batch_samples = data.size(0)  # get the batch size
            data = data.view(batch_samples, data.size(1),
                             -1)  # flatten the image pixels except for batch and channel dimensions
            mean += data.mean(2).sum(0)  # accumulate the sum of means for each channel
            std += data.std(2).sum(0)  # accumulate the sum of stds for each channel
            nb_samples += batch_samples  # accumulate the total number of samples

        # Calculate the final mean and std by dividing by the total number of samples
        mean /= nb_samples
        std /= nb_samples

        self.transform = transforms.Compose([
            self.initial_transform,
            transforms.Normalize(mean=mean, std=std)
        ])

        mean_str = '[' + ', '.join([f'{m:.4f}' for m in mean.tolist()]) + ']'
        std_str = '[' + ', '.join([f'{s:.4f}' for s in std.tolist()]) + ']'
        print(f" - Normalized dataset:\n  - Mean: {mean_str}\n  - Standard deviation: {std_str}")
        # mean=[0.4673, 0.4897, 0.4125], std=[0.1670, 0.1398, 0.1845]
