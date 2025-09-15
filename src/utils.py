import pyedflib
import numpy as np
from scipy.signal import resample_poly
from pathlib import Path

import matplotlib.pyplot as plt

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_edf_signals(edf_path):
    with pyedflib.EdfReader(edf_path) as f:
        n = f.signals_in_file
        ch_names = f.getSignalLabels()
        fs_list = np.array([f.getSampleFrequency(i) for i in range(n)], dtype=float)
        signals = [f.readSignal(i).astype(float) for i in range(n)]
    return ch_names, fs_list, signals


def save_feature_matrix_npz(out_npz: Path, X, meta: dict):
    ensure_dir(out_npz.parent)
    payload = {"X": np.asarray(X, dtype=np.float32)}
    if meta:
        for k, v in meta.items():
            if isinstance(v, (str, int, float, np.integer, np.floating)):
                payload[k] = v
    np.savez_compressed(out_npz, **payload)
    return out_npz


def load_feature_matrix_npz(path: str):
    d = np.load(path, allow_pickle=False)
    return d["X"]


def robust_normalize(spectrogram, p=5.0, method="z-score"):
    if method == "z-score":
        return _robst_z_score_normalization(spectrogram, p)  # <-- return!
    elif method == "min-max":
        raise NotImplementedError("min-max not implemented yet.")
    else:
        raise ValueError(f"Unknown method: {method}")


def _robst_z_score_normalization(x, p):
    # work on a copy so we don't mutate x while estimating stats
    truncated = _truncate_signals(x.copy(), p, 100. - p)
    robust_mean = np.mean(truncated)
    robust_std  = np.std(truncated)
    if robust_std == 0 or not np.isfinite(robust_std):
        robust_std = 1.0
    return (x - robust_mean) / robust_std


def _truncate_signals(vec, min_percentile, max_percentile):
    min_value = np.percentile(vec, min_percentile)
    max_value = np.percentile(vec, max_percentile)
    vec[vec < min_value] = min_value
    vec[vec > max_value] = max_value
    return vec


def _resample_to(signal, sf_in, sf_out):
    if sf_in == sf_out:
        return signal, sf_in
    # resample_poly handles anti-aliasing
    # choose integers for up/down to keep precision
    from math import gcd
    g = gcd(int(sf_out), int(sf_in))
    up, down = int(sf_out // g), int(sf_in // g)
    return resample_poly(signal, up, down), float(sf_out)


def plot_spectrogram(f, t, S, title=None, out_png:Path = None, show=False):
    """Plot a single spectrogram (F x T)."""
    # pick a stable color scale so different plots are comparable
    vmin, vmax = np.percentile(S, [5, 95])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = None, None

    plt.figure(figsize=(8, 4))
    plt.pcolormesh(t, f, S, shading="auto", vmin=vmin, vmax=vmax)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title or "Spectrogram")
    cbar = plt.colorbar()
    cbar.set_label("Power (normalized)")
    plt.tight_layout()
    if out_png is not None:
        plt.savefig(out_png, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
