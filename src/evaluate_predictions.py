import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import medfilt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from src.prediction_utils import decode_stages

CLASS_ORDER = ["Wake", "NREM", "REM"]  # adjust if needed
INT_TO_STAGE = {0:"Wake", 1:"NREM", 2:"REM"}


def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=None, title=None):
    """
    Seaborn heatmap confusion matrix.
      normalize: None, 'true', 'pred', or 'all' (same as sklearn)
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    fmt = ".2f" if normalize else "d"
    vmax = 1.0 if normalize else cm.max()

    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues", vmin=0, vmax=vmax,
        xticklabels=labels, yticklabels=labels, cbar=True, square=True
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ttl = title or "Confusion matrix"
    if normalize:
        ttl += f" (normalized: {normalize})"
    ax.set_title(ttl)
    plt.tight_layout()
    plt.show()


def smooth_median(y_pred, kernel=5):
    if kernel % 2 == 0:
        kernel += 1  # must be odd
    sm = medfilt(y_pred, kernel_size=kernel)
    return sm

def smooth_by_animal(y, groups, width=5):
    """Apply majority smoothing within each animal (no leakage across animals)."""
    y = np.asarray(y)
    g = np.asarray(groups)
    y_sm = y.copy()
    for animal in np.unique(g):
        idx = np.flatnonzero(g == animal)
        y_sm[idx] = smooth_median(y[idx], kernel=width)
    return y_sm




def make_val_df(groups_va, y_va_enc, y_pred, y_pred_smoothed, epoch_len_sec=4.0):
    df = pd.DataFrame({
        "animal_id": np.asarray(groups_va),
        "label":     np.asarray(y_va_enc, dtype=int),
        "pred":      np.asarray(y_pred, dtype=int),
        "pred_sm":   np.asarray(y_pred_smoothed, dtype=int),
    })
    # per-animal epoch index and time (s) from the start of that animal’s validation block
    df["epoch_idx"] = df.groupby("animal_id").cumcount()
    df["time_s"]    = df["epoch_idx"] * float(epoch_len_sec)
    return df

def subset_window(df: pd.DataFrame,
                  animal_id: str,
                  start_h: float = 0.0,
                  duration_h: float = 2.0):
    """
    Returns a dict with arrays for a time window for one animal.
    df must have: animal_id, time_s, label, pred, pred_sm, epoch_idx
    """
    t0 = start_h * 3600.0
    t1 = t0 + duration_h * 3600.0
    sel = (df["animal_id"] == animal_id) & (df["time_s"] >= t0) & (df["time_s"] < t1)
    sub = df.loc[sel].sort_values("time_s")
    return {
        "times_s": sub["time_s"].to_numpy(),
        "y_true":  sub["label"].to_numpy(),
        "y_pred":  sub["pred"].to_numpy(),
        "y_pred_sm": sub["pred_sm"].to_numpy(),
        "epoch_idx": sub["epoch_idx"].to_numpy(),
    }


def plot_hypnogram_window(times_s, y_true=None, y_pred=None, y_pred_smooth=None, title=""):
    xh = times_s / 3600.0
    plt.figure(figsize=(16, 4))
    if y_true is not None:
        plt.step(xh, y_true, where="post", alpha=0.6, label="Ground truth")
    if y_pred is not None:
        plt.step(xh, y_pred, where="post", alpha=0.4, label="Predicted")
    if y_pred_smooth is not None:
        plt.step(xh, y_pred_smooth, where="post", lw=2, alpha=0.4, label="Pred (smoothed)")

    # Put Wake at the top (invert y-axis)
    plt.yticks([0,1,2], [INT_TO_STAGE[0], INT_TO_STAGE[1], INT_TO_STAGE[2]])
    plt.ylim(2.5, -0.5)

    plt.xlabel("Time (hours)")
    plt.ylabel("Stage")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
