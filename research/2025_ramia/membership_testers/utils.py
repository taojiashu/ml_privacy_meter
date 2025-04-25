import pdb

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import scipy
import itertools

from dataset_loaders.base_datasets import PrivateDataset


def compute_loss(
    model, data, loss_function, device, aggregate=True, unknown_label=False
):
    model.eval()
    model.to(device)

    if aggregate:
        batch_size = 128
    else:
        batch_size = 1

    if isinstance(data, Dataset):
        data = DataLoader(data, batch_size=batch_size, shuffle=False)
    elif isinstance(data, DataLoader):
        pass
    elif isinstance(data, list) or isinstance(data, np.ndarray):
        data = np.array(data)
        # if data.ndim == 0:
        #     data = data.reshape(-1)
        data = DataLoader(
            PrivateDataset(data[..., :-1], data[..., -1]),
            batch_size=batch_size,
            shuffle=False,
        )
    elif isinstance(data, tuple):
        pdb.set_trace()
        data = DataLoader(
            PrivateDataset(data[0].permute(1, 2, 0), data[1]),
            batch_size=batch_size,
            shuffle=False,
        )

    if unknown_label:
        predictions = []
        with torch.no_grad():
            for inputs, _ in data:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted_classes = torch.max(
                    outputs, 1
                )  # Get the index of the max log-probability
                predictions.extend(predicted_classes.tolist())

        # Count the occurrences of each class in the predictions
        count = Counter(predictions)
        majority_class, _ = count.most_common(1)[0]  # Get the most common class

    if aggregate:
        loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data):
                data, target = data.to(device), target.to(device)
                output = model(data)
                if unknown_label:
                    # target = torch.tensor([majority_class] * len(target), dtype=torch.long).to(device)
                    target = torch.ones_like(target, dtype=torch.long) * majority_class
                    target = target.to(device)
                loss += loss_function(output, target).sum().item()  # sum up batch loss
        loss /= batch_idx + 1
    else:
        loss = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data):
                data, target = data.to(device), target.to(device)
                output = model(data)
                if unknown_label:
                    # target = torch.tensor([majority_class] * len(target), dtype=torch.long).to(device)
                    target = torch.ones_like(target, dtype=torch.long) * majority_class
                    target = target.to(device)
                loss.append(loss_function(output, target).item())
            loss = np.array(loss)
    return loss


def get_logit(model, data, device, rescale=False, unknown_label=False):
    model.eval()

    if isinstance(data, Dataset):
        data = DataLoader(data, batch_size=1, shuffle=False)
    elif isinstance(data, DataLoader):
        pass
    elif (
        isinstance(data, list)
        or isinstance(data, np.ndarray)
        or isinstance(data, tuple)
    ):
        data = np.array(data)
        # if data.ndim == 0:
        #     data = data.reshape(-1)
        data = DataLoader(
            PrivateDataset(data[..., :-1], data[..., -1]), batch_size=1, shuffle=False
        )

    logits = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data):
            data, target = data.to(device), target.to(device)
            # output = F.softmax(model(data))
            if not unknown_label:
                # pdb.set_trace()
                if target.ndim > 1:
                    output = F.sigmoid(model(data))
                    # pdb.set_trace()
                    logits.append((output * target.long()).mean().cpu().numpy())
                else:
                    output = F.softmax(model(data))
                    logits.append(output[:, target.long()].cpu().numpy())
            else:
                output = F.softmax(model(data))
                pred = output.argmax(dim=-1)
                logits.append(output[:, pred].cpu().numpy())
    logits = np.array(logits)
    if rescale:
        logits = np.log(logits + 1e-23) - np.log(1 - logits + 1e-23)
    return logits.reshape(-1)


def get_rmia_score_arrays_from_loss_arrays(
    loss_arrays, split_matrix, offline=False, a=None
):
    assert (
        len(loss_arrays) == split_matrix.shape[0]
    ), "The number of loss arrays should be equal to the number of models."
    assert all(
        item.shape[0] == split_matrix.shape[1] for item in loss_arrays
    ), "All items in the loss_arrays should have the same number of data points."

    target_loss = loss_arrays[0]
    target_prob = np.exp(-target_loss)

    in_prob = np.zeros_like(target_prob)
    in_prob_count = np.zeros_like(target_prob)
    out_prob = np.zeros_like(target_prob)
    out_prob_count = np.zeros_like(target_prob)

    from dataset_loaders.utils import get_data_index_for_model_from_split

    for i in range(1, len(loss_arrays)):
        prob = np.exp(-loss_arrays[i])
        in_indices = get_data_index_for_model_from_split(split_matrix, i)
        in_prob[in_indices] += prob[in_indices]
        in_prob_count[in_indices] += 1

        out_indices = np.where(split_matrix[i] == 0)[0]
        out_prob[out_indices] += prob[out_indices]
        out_prob_count[out_indices] += 1

    assert np.all(
        in_prob_count + out_prob_count == len(loss_arrays) - 1
    ), "{} entries have in_prob_count + out_prob_count != len(loss_arrays) - 1.".format(
        np.sum(in_prob_count + out_prob_count != len(loss_arrays) - 1)
    )
    in_prob /= in_prob_count
    out_prob /= out_prob_count

    if offline:
        in_prob = out_prob * a + (1 - a)
        # in_prob = np.sqrt(out_prob)

    rmia_scores = target_prob / (0.5 * in_prob + 0.5 * out_prob)
    return rmia_scores


def get_rmia_score_dict_from_loss_dicts(
    loss_dicts, split_matrix, offline=False, a=None
):
    assert (
        len(loss_dicts) == split_matrix.shape[0]
    ), "The number of loss dicts should be equal to the number of models."
    assert all(
        isinstance(item, dict) for item in loss_dicts
    ), "All items in the loss_dicts should be dictionaries."

    target_loss = loss_dicts[0]
    target_prob = {key: np.exp(-np.array(value)) for key, value in target_loss.items()}

    in_prob = {key: np.zeros_like(value) for key, value in target_prob.items()}
    in_prob_count = {key: 0 for key, value in target_prob.items()}
    out_prob = {key: np.zeros_like(value) for key, value in target_prob.items()}
    out_prob_count = {key: 0 for key, value in target_prob.items()}

    from dataset_loaders.utils import get_data_index_for_model_from_split

    for i in range(1, len(loss_dicts)):
        prob = {key: np.exp(-np.array(value)) for key, value in loss_dicts[i].items()}
        in_indices = get_data_index_for_model_from_split(split_matrix, i) + 1
        for key in in_indices:
            in_prob[key] += prob[key]
            in_prob_count[key] += 1
        out_indices = np.where(split_matrix[i] == 0)[0] + 1
        for key in out_indices:
            out_prob[key] += prob[key]
            out_prob_count[key] += 1

    for key in in_prob.keys():
        if in_prob_count[key] == 0:
            in_prob_count[key] = np.zeros_like(out_prob_count[key])
            # raise ValueError("in_prob_count should not be zero.")
        elif not np.isfinite(in_prob[key]).any():
            raise ValueError("in_prob should not be infinite.")
        else:
            # pdb.set_trace()
            in_prob[key] /= in_prob_count[key]
        out_prob[key] /= out_prob_count[key]

    if offline:
        # in_prob = {key: np.sqrt(value) for key, value in out_prob.items()}
        in_prob = {key: out_prob[key] * a + (1 - a) for key in out_prob.keys()}

    rmia_scores = {
        key: target_prob[key] / (0.5 * in_prob[key] + 0.5 * out_prob[key] + 1e-23)
        for key in target_prob.keys()
    }
    return rmia_scores


def get_rmia_scores_for_all_loss_arrays(
    loss_arrays, split_matrix, offline=False, a=None
):
    rmia_scores = []

    split = split_matrix.copy()
    loss = loss_arrays.copy()

    for i in range(len(loss)):
        if i > 0:
            target_loss = loss.pop(i)
            loss.insert(0, target_loss)
            # Moving the split matrix row to the top
            split = np.concatenate(
                (
                    split[i].reshape(1, -1),
                    split[:i],
                    split[i + 1 :],
                ),
                axis=0,
            )

        rmia_scores.append(
            get_rmia_score_arrays_from_loss_arrays(loss, split, offline=offline, a=a)
        )
    return np.array(rmia_scores)
