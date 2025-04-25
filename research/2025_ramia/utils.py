import torch
import pandas as pd
import numpy as np


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def save_fpr_tpr_csv(fpr, tpr, filename, subsample=True):
    """
    Save FPR and TPR values to a CSV file.

    Parameters:
    - fpr: list of numpy arrays (FPR values)
    - tpr: list of numpy arrays (TPR values)
    - filename: str, the path to save the CSV file
    """
    # Convert lists of arrays to a single DataFrame
    if (
        isinstance(fpr, list)
        and all(isinstance(arr, np.ndarray) for arr in fpr)
        and isinstance(tpr, list)
        and all(isinstance(arr, np.ndarray) for arr in tpr)
    ):
        for i in range(len(fpr)):
            fpr[i] = fpr[i].reshape(-1)
            tpr[i] = tpr[i].reshape(-1)
            if subsample:
                # Ensure first and last elements are included, and subsample in between
                fpr[i] = np.concatenate(([fpr[i][0]], fpr[i][1:-1:len(fpr[i]) // 100], [fpr[i][-1]]))
                tpr[i] = np.concatenate(([tpr[i][0]], tpr[i][1:-1:len(tpr[i]) // 100], [tpr[i][-1]]))
            df = pd.DataFrame({"FPR": fpr[i], "TPR": tpr[i]})
            if filename.endswith(".csv"):
                filename_new = filename.split(".csv")[0] + f"_{i}.csv"
            else:
                filename_new = filename + f"_{i}.csv"
            df.to_csv(filename_new, index=False)
    else:
        if subsample:
            # Ensure first and last elements are included, and subsample in between
            fpr = np.concatenate(([fpr[0]], fpr[1:-1:len(fpr) // 100], [fpr[-1]]))
            tpr = np.concatenate(([tpr[0]], tpr[1:-1:len(tpr) // 100], [tpr[-1]]))
        df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        df.to_csv(filename, index=False)
