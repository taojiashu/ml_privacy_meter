from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler

import numpy as np
from argparse import ArgumentParser
from dataset_loaders import CelebADatasetLoader
from train_model import *
from facial_attribute_cnn import FacialAttrCNN


def combine_dict_lists(dict_of_lists):
    """
    Combines all list values from a dictionary into a single list.

    Parameters:
    - dict_of_lists (dict): A dictionary where each value is expected to be a list.

    Returns:
    - list: A list containing all the elements from the dictionary's list values.
    """
    combined_list = []
    for value_list in dict_of_lists.values():
        combined_list.extend(value_list)
    return combined_list


if __name__ == "__main__":
    # Initialize parser
    parser = ArgumentParser()
    parser.add_argument("--num_models", type=int, default=4)
    parser.add_argument("--model_id", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    if args.model_id >= args.num_models:
        raise ValueError("Model ID must be less than the number of models")

    # Initialize device
    device = torch.device(
        "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu"
    )

    # Load CelebA
    celeba_all = celeba = CelebADatasetLoader(
        load_from_disk=True, dataset_path="datasets/celeba"
    ).load_all_original_data()

    # Load specific id
    try:
        id = np.load("datasets/celeba/model_{}_id.npy".format(args.model_id))
    except FileNotFoundError:
        id_dict = defaultdict(list)
        for i in tqdm(range(len(celeba_all))):
            id = celeba_all[i][1][-1]
            id_dict[id.item()].append(i)
        id_0 = []
        id_1 = []
        id_2 = []
        id_3 = []

        for id, values in id_dict.items():
            if id > 0 and id <= 5089:
                id_0.extend(values[:len(values) // 2])
                id_1.extend(values[len(values) // 2:])
            elif id > 5089 and id <= 10177:
                id_2.extend(values[:len(values) // 2])
                id_3.extend(values[len(values) // 2:])

            train_loader = torch.utils.data.DataLoader(
                celeba_all,
                batch_size=128,
                sampler=SubsetRandomSampler(id),
                num_workers=2,
            )
        np.save("datasets/celeba/model_0_id.npy", np.array(id_0))
        np.save("datasets/celeba/model_1_id.npy", np.array(id_1))
        np.save("datasets/celeba/model_2_id.npy", np.array(id_2))
        np.save("datasets/celeba/model_3_id.npy", np.array(id_3))
        id = np.load("datasets/celeba/model_{}_id.npy".format(args.model_id))

    # Model training
    model = FacialAttrCNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_multilabel_model(
        model,
        train_loader,
        100,
        criterion,
        optimizer,
        device=device,
    )

    # Save model
    torch.save(
        model.state_dict(), "saved_models/celeba/model_{}.pt".format(args.model_id)
    )
