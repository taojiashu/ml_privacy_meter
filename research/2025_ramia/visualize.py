from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import numpy as np

from utils import save_fpr_tpr_csv


def plot_roc(fpr, tpr, title, save_path=None):
    """
    Plots the ROC curve using provided FPR and TPR and displays the AUC score in the legend.

    Parameters:
    - fpr: array-like, shape = [n] False Positive Rates
    - tpr: array-like, shape = [n] True Positive Rates
    - title: str, title of the plot
    - save_path: str, path to save the plot

    Returns:
    - None
    """
    roc_auc = auc(fpr, tpr)  # Calculate AUC

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path + ".png")
        save_fpr_tpr_csv(fpr=[fpr], tpr=[tpr], filename=save_path + ".csv")

    plt.show(block=False)


def plot_multiple_roc_curves(fpr, tpr, title, save_path=None):
    """
    Plots multiple ROC curves on a single plot with both linear and logarithmic x-axis scales.
    Displays each ROC curve's AUC in the legend.

    Parameters:
    - fpr: list of numpy arrays, where each array contains False Positive Rates for a model
    - tpr: list of numpy arrays, where each array contains True Positive Rates for a model
    - title: str, title of the plot
    - save_path: str, path to save the plot

    Returns:
    - None
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))

    ax[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.0])
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    ax[0].set_title(f"{title} (Linear Scale)")

    ax[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax[1].set_xlim([1e-5, 1.0])  # Set lower limit to avoid log(0)
    ax[1].set_ylim([1e-5, 1.0])
    ax[1].set_xlabel("FPR")
    ax[1].set_ylabel("TPR")
    ax[1].set_title(f"{title} (Log Scale)")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")

    # Check if fpr and tpr are lists of numpy arrays
    if (
        isinstance(fpr, list)
        and all(isinstance(arr, np.ndarray) for arr in fpr)
        and isinstance(tpr, list)
        and all(isinstance(arr, np.ndarray) for arr in tpr)
    ):
        for i, (fpr_arr, tpr_arr) in enumerate(zip(fpr, tpr)):
            roc_auc = auc(fpr_arr, tpr_arr)
            ax[0].plot(
                fpr_arr, tpr_arr, lw=2, label=f"ROC curve {i + 1} (AUC = {roc_auc:.4f})"
            )
            ax[1].plot(
                fpr_arr, tpr_arr, lw=2, label=f"ROC curve {i + 1} (AUC = {roc_auc:.4f})"
            )
    else:  # Fallback for single ROC curve (though this case assumes lists of arrays as input)
        raise ValueError("fpr and tpr must be lists of numpy arrays.")

    # ax[0].legend(loc="lower right")
    ax[0].legend(fontsize=8)
    # ax[1].legend(loc="lower right")
    ax[1].legend(fontsize=8)

    if save_path:
        plt.savefig(save_path + ".png")
        save_fpr_tpr_csv(fpr=fpr, tpr=tpr, filename=save_path + ".csv")

    plt.show(block=False)


def find_tpr_at_fpr_threshold(fpr, tpr, threshold):
    """
    Finds the largest FPR value that is less than or equal to a given threshold and returns the corresponding TPR value.

    Parameters:
    - fpr (numpy.ndarray): Array of false positive rates.
    - tpr (numpy.ndarray): Array of true positive rates.
    - threshold (float): The FPR threshold to find the TPR at.

    Returns:
    - tuple: A tuple containing the largest FPR below the threshold and the corresponding TPR value.
    """
    # Filter indices where FPR is below or equal to the threshold
    valid_indices = np.where(fpr <= threshold)[0]

    if not valid_indices.size:
        return None, None  # or raise an Exception if preferred

    # Find the index of the maximum FPR that is still below the threshold
    max_fpr_index = valid_indices[
        -1
    ]  # Last element in valid_indices will be the largest below the threshold

    # Return the maximum FPR below the threshold and the corresponding TPR value
    return fpr[max_fpr_index], tpr[max_fpr_index]
