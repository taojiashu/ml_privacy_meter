from abc import abstractmethod, ABC

import numpy as np

import torch
from torch.utils.data import Dataset


class BaseDatasetLoader(ABC):
    def __init__(self, load_from_disk=False, dataset_path=None):
        self.load_from_disk = load_from_disk
        self.dataset_path = dataset_path
        if load_from_disk and self.dataset_path is None:
            raise ValueError("Dataset path is None")

        # self.training_set, self.test_set, self.population_set = self.load_data()

    @abstractmethod
    def load_data(self):
        pass


# class RangeDatasetLoader(ABC):
#     def __init__(self, load_from_disk=False, dataset_path=None, distance_function=None, range=None, copies=1):
#         self.load_from_disk = load_from_disk
#         self.dataset_path = dataset_path
#         self.distance_function = distance_function
#         self.range = range
#         self.copies = copies

#         self.training_set, self.test_set, self.population_set = self.load_data()

#     @abstractmethod
#     def get_range_dataset(self):
#         """
#         Turns a normal dataset into a range dataset.
#         Args:
#             distance_function: The distance function to use for the range dataset.
#             range: The radius to use for the range dataset.
#             copies: The number of perturbed samples for each true sample.
#         Returns:
#             A range dataset.
#         """
#         pass


class PrivateDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        labels = np.round(labels).clip(min=0)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

        if self.labels.ndim == 0:
            self.labels = self.labels.view(-1)
        n = len(self.labels)
        
        if self.features.ndim == 0:
            self.features = self.features.view(n, -1)
        elif self.features.ndim == 1:
            self.features = self.features.view(n, -1)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.features[idx]), self.labels[idx]
        return self.features[idx], self.labels[idx]
