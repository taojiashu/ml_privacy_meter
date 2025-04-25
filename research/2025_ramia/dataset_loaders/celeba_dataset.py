import os
import typing

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CelebA

from .base_datasets import BaseDatasetLoader


class CelebADataset(Dataset):
    def __init__(self, data, attr, identity, transform=None):
        if type(data) == list:
            self.data = torch.stack(data)
        else:
            self.data = torch.tensor(data)
        if type(attr) == list:
            self.attr = torch.stack(attr).to(torch.float32)
        else:
            self.attr = torch.tensor(attr, dtype=torch.float32)
        if type(identity) == list:
            self.identity = torch.stack(identity).to(torch.float32)
        else:
            self.identity = torch.tensor(identity, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        attr = self.attr[idx]
        # identity = self.identity[idx]

        if self.transform:
            sample = self.transform(sample)

        # return sample, attr, identity
        return sample, attr
    
    def get_all_attributes(self, idx):
        sample = self.data[idx]
        attr = self.attr[idx]
        identity = self.identity[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, attr, identity
    

class CelebADatasetLoader(BaseDatasetLoader):
    def __init__(self, load_from_disk=False, dataset_path="datasets/celeba/"):
        super().__init__(load_from_disk, dataset_path)
        self.load_from_disk = load_from_disk
        self.dataset_path = dataset_path
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        self.training_set, self.test_set, self.population_set, self.nonmembers_set = self.load_data()

    def load_original_data(self):
        try:
            celeba_train = CelebA(root='datasets', split='train', target_type=["attr", "identity"], transform=self.transform, download=False)
            celeba_valid = CelebA(root='datasets', split='valid', target_type=["attr", "identity"], transform=self.transform, download=False)
            celeba_test = CelebA(root='datasets', split='test', target_type=["attr", "identity"], transform=self.transform, download=False)
        except:
            celeba_train = CelebA(root='datasets', split='train', target_type=["attr", "identity"], transform=self.transform, download=True)
            celeba_valid = CelebA(root='datasets', split='valid', target_type=["attr", "identity"], transform=self.transform, download=True)
            celeba_test = CelebA(root='datasets', split='test', target_type=["attr", "identity"], transform=self.transform, download=True)

        return celeba_train, celeba_valid, celeba_test
    
    def load_all_original_data(self):
        try:
            celeba_all = CelebA(root='datasets', split='all', target_type=["attr", "identity"], transform=self.transform, download=False)
        except:
            celeba_all = CelebA(root='datasets', split='all', target_type=["attr", "identity"], transform=self.transform, download=True)
        return celeba_all
    
    def load_data(self) -> typing.Tuple[Dataset, Dataset, Dataset, Dataset]:
        if self.load_from_disk:
            training_set_path = os.path.join(self.dataset_path, "train.pt")
            test_set_path = os.path.join(self.dataset_path, "test.pt")
            population_set_path = os.path.join(self.dataset_path, "population.pt")
            nonmembers_set_path = os.path.join(self.dataset_path, "nonmembers.pt")
            return torch.load(training_set_path), torch.load(test_set_path), torch.load(population_set_path), torch.load(nonmembers_set_path)
        else:
            celeba = self.load_all_original_data()

            # Identities in the last 50% will be add to non-members set
            nonmembers_data = []
            nonmembers_attr = []
            nonmembers_identity = []

            members_data = []
            members_attr = []
            members_identity = []

            for i in range(len(celeba)):
                if celeba.identity[i] < 10177 * 0.5:
                    members_data.append(celeba[i][0])
                    members_identity.append(celeba.identity[i])
                    members_attr.append(celeba.attr[i])
                else:
                    nonmembers_data.append(celeba[i][0])
                    nonmembers_identity.append(celeba.identity[i])
                    nonmembers_attr.append(celeba.attr[i])
            
            # Save the nonmembers dataset and the population dataset
            nonmembers_data, population_data, nonmembers_attr, population_attr, nonmembers_identity, population_identity = \
                train_test_split(nonmembers_data, nonmembers_attr, nonmembers_identity, test_size=0.5, random_state=42)        

            nonmembers_attr = torch.stack(nonmembers_attr)
            nonmembers_identity = torch.stack(nonmembers_identity)
            nonmembers_set = CelebADataset(nonmembers_data, nonmembers_attr, nonmembers_identity)
            torch.save(nonmembers_set, "datasets/celeba/nonmembers.pt")

            # Split the members dataset into training, population and test set
            members_data_train, members_data_test, members_identity_train, members_identity_test, members_attr_train, members_attr_test = \
                train_test_split(members_data, members_identity, members_attr, test_size=0.5, random_state=42)
            
            # Save the datasets
            training_set = CelebADataset(members_data_train, members_attr_train, members_identity_train)
            test_set = CelebADataset(members_data_test, members_attr_test, members_identity_test)
            population_set = CelebADataset(population_data, population_attr, population_identity)
            torch.save(training_set, "datasets/celeba/train.pt")
            torch.save(test_set, "datasets/celeba/test.pt")
            torch.save(population_set, "datasets/celeba/population.pt")
            
            return training_set, test_set, population_set, nonmembers_set