import numpy as np
import pandas as pd
import mne
from typing import Optional, Tuple

from scipy.signal import spectrogram
from lspopt import spectrogram_lspopt

from src.make_spectrograms import get_spectrogram
from src.utils import robust_normalize


ARTIFACT_CODES = {"1", "2", "3"} 
STAGE_MAP = {
    "w": "Wake", "wake": "Wake",
    "n": "NREM", "nr": "NREM", "nrem": "NREM",
    "r": "REM",  "rem": "REM"
    }

PRIORITY = {"REM": 3, "NREM": 2, "Wake": 1, "Unknown": 0}
STAGE_TO_INT = {"Wake": 0, "NREM": 1, "REM": 2}


def norm_stage(x):
    x = str(x).strip().lower()
    return STAGE_MAP.get(x, "Unknown")

def is_artifact(a, b):
    return (str(a).strip() in ARTIFACT_CODES) or (str(b).strip() in ARTIFACT_CODES)

def consensus(a, b, consensus_rule) -> str:
        a, b = norm_stage(a), norm_stage(b)
        if consensus_rule == "agree":
            return a if a == b else "Unknown"
        if consensus_rule == "rater1":
            return a
        if consensus_rule == "rater2":
            return b
        # default: "priority"
        if a == b: return a
        if a == "Unknown": return b
        if b == "Unknown": return a
        return a if PRIORITY[a] >= PRIORITY[b] else b

def get_annotations(scoring_path: str, consensus_rule: str = "priority") -> pd.DataFrame:
    """
    Load and fuse annotations from two raters.
    consensus_rule: "priority" | "agree" | "rater1" | "rater2"
    Returns a DataFrame with columns: 'stage' (str), 'artifact' (bool)
    """
    lab = pd.read_csv(scoring_path, header=None, sep=r"[;,]", engine="python")  
    s1 = lab.iloc[:, -2].astype(str)
    s2 = lab.iloc[:, -1].astype(str)

    labels = pd.DataFrame({
        "stage":    [consensus(a, b, consensus_rule) for a, b in zip(s1, s2)],
        "artifact": [is_artifact(a, b) for a, b in zip(s1, s2)],
    })
    return labels

def concat_every_k_rows(X: np.ndarray, k: int = 4) -> np.ndarray:
    """
    Concatenate every k consecutive rows horizontally:
    (T, F) -> (T//k, k*F)
    """
    T, F = X.shape
    T_used = (T // k) * k
    if T_used == 0:
        raise ValueError("Not enough rows to form one epoch.")
    X_used = X[:T_used]
    return X_used.reshape(-1, k, F).reshape(-1, k * F)

def align_labels(y: np.ndarray, n_epochs: int) -> np.ndarray:
    y = np.asarray(y)
    if len(y) < n_epochs:
        raise ValueError(f"y too short: {len(y)} < {n_epochs}")
    return y[:n_epochs]

def make_epoch_dataframe(animal_id: str, X_epochs: np.ndarray, y_epochs: np.ndarray, epoch_len_sec: int = 4) -> pd.DataFrame:
    """
    Create a tidy per-epoch DF with metadata and features.
    'features' is stored as a numpy array per row (object column).
    """
    E, D = X_epochs.shape
    df_rec = pd.DataFrame({
        "animal_id": animal_id,
        "epoch_idx": np.arange(E, dtype=np.int32),
        "start_sec": (np.arange(E, dtype=np.int64) * epoch_len_sec),
        "label": y_epochs
    })
    # store features as arrays (object dtype); efficient in-memory and fine for pickle/parquet
    df_rec["features"] = [row.astype(np.float32, copy=False) for row in X_epochs]
    return df_rec



def preprocess_eeg(raw_signal: np.ndarray,
                   sampling_freq_hz: float,
                   win_sec: float = 1.0,
                   overlap_sec: float = 0.0,
                   *,
                   multi_taper: bool = True,
                   low_cut_hz: float = 0.5,
                   high_cut_hz: float = 60.0,
                   notch_low_cut_hz: float = 45.0,
                   notch_high_cut_hz: float = 55.0,
                   normalize_method: str = "z-score",
                   normalize_p: float = 5.0
                   ):


    """    Preprocess raw EEG signal by applying bandpass filter and notch filter.
    Args:
        raw_signal (np.ndarray): Raw EEG signal.
        sampling_freq_hz (int): Sampling frequency in Hz.
        low_cut_hz (float): Low cut frequency for bandpass filter.
        high_cut_hz (float): High cut frequency for bandpass filter.
        notch_low_cut_hz (float): Low cut frequency for notch filter.
        notch_high_cut_hz (float): High cut frequency for notch filter.
        multi_taper (bool): Whether to use multi-taper method for spectrogram.
        nperseg (int): Length of each segment for spectrogram.
        noverlap (int): Number of points to overlap between segments for spectrogram.
    Returns:
        frequencies (np.ndarray): Frequencies of the spectrogram.
        time (np.ndarray): Time points of the spectrogram.
        spectrogram (np.ndarray): Preprocessed spectrogram.

    """

    # Get spectrogram
    nperseg = max(8, int(round(sampling_freq_hz * win_sec)))  # at least a few sample
    noverlap = int(round(sampling_freq_hz * overlap_sec))
    f, t, S = get_spectrogram(raw_signal, sampling_freq_hz, multi_taper=multi_taper,
                                nperseg=nperseg, noverlap=noverlap)
    
    # Filter overall low and high frequencies
    nyq = sampling_freq_hz / 2.
    high_cut_eff = min(high_cut_hz, nyq - 1e-6)
    freq_mask = (f >= low_cut_hz) & (f <= high_cut_eff)
    f = f[freq_mask]
    S = S[freq_mask, :]

    # Apply notch filter
    notch_mask = (f < notch_low_cut_hz) | (f > notch_high_cut_hz)
    f = f[notch_mask]
    S = S[notch_mask, :]

    # log transform to make more Gaussian
    S = np.log1p(S)

    # normalize data
    S = robust_normalize(S, p=normalize_p, method=normalize_method)

    return f, t, S



def preprocess_recording_and_label(
    edf_path: str,
    scoring_path: str,
    *,
    epoch_len: float = 4.0,
    line_freq=50.0,
    line_notch_width: float = 6.0,   # None = no notch; else width in Hz  
    eeg_band: Tuple[float, float] = (0.5, 45.0),
    emg_band: Tuple[float, float] = (10.0, 50.0),
    resample_hz: Optional[float] = None,    # e.g. 200; None = keep original
    consensus_rule: str = "priority",       # "priority" | "agree" | "rater1" | "rater2"
):
    """
    Returns
    -------
    X : np.ndarray, shape (N, C, T)
        Cleaned 4-s epochs (EEG1, EEG2, EMG) as time series.
    y : np.ndarray, shape (N,)
        Integer labels aligned to X, with map {'Wake':1,'NREM':2,'REM':3}.
    meta : dict
        Metadata (sfreq, indices kept, counts, channel names, etc.).
    """

    # -------- 1) Load + fuse labels --------
    lab = pd.read_csv(scoring_path, header=None, sep=r"[;,]", engine="python")  
    s1 = lab.iloc[:, -2].astype(str)
    s2 = lab.iloc[:, -1].astype(str)

    labels = pd.DataFrame({
        "stage":    [consensus(a, b, consensus_rule) for a, b in zip(s1, s2)],
        "artifact": [is_artifact(a, b) for a, b in zip(s1, s2)],
    })

    # -------- 2) Read EDF + channel typing --------
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.rename_channels(lambda s: s.strip())

    eeg_like = [ch for ch in raw.ch_names if "EEG" in ch.upper()]
    emg_like = [ch for ch in raw.ch_names if "EMG" in ch.upper()]
    if not eeg_like:
        raise ValueError("No EEG-like channels found (need names containing 'EEG').")
    if not emg_like:
        raise ValueError("No EMG-like channels found (need names containing 'EMG').")

    raw.set_channel_types({ch: "eeg" for ch in eeg_like})
    raw.set_channel_types({ch: "emg" for ch in emg_like})
    #raw.set_eeg_reference("average")

    if resample_hz is not None:
        raw.resample(resample_hz)

    # -------- 3) Filters (notch -> band-pass) --------
    if line_notch_width is not None:
        raw.notch_filter(
            freqs=line_freq,    # e.g., [50]
            picks="eeg",
            method="iir",       # good at low sfreq; or use FIR if you prefer
            notch_widths=line_notch_width,
            verbose=False
    )
    
    raw.filter(*eeg_band, picks="eeg", verbose=False)
    raw.filter(*emg_band, picks="emg", verbose=False)

    # -------- 4) Sample-accurate 4-s epoching --------
    sf = float(raw.info["sfreq"])
    spe = int(round(epoch_len * sf))     # samples per epoch
    elen = spe / sf                      # exact seconds on the sample grid
    n_possible = raw.n_times // spe
    n_labels = len(labels)
    n_fit = int(min(n_possible, n_labels))
    if n_fit <= 0:
        raise ValueError("No full epochs fit the recording with the given epoch_len.")

    stop_sample = n_fit * spe
    raw.crop(tmin=0.0, tmax=(stop_sample - 1) / sf)

    epochs = mne.make_fixed_length_epochs(
        raw, duration=elen, overlap=0.0, preload=True, reject_by_annotation=False, verbose=False
    )

    # -------- 5) Align labels, mask, arrays --------
    labels = labels.iloc[:len(epochs)].reset_index(drop=True)
    stage_ok = labels["stage"].isin(STAGE_TO_INT)
    good_mask = (stage_ok) & (~labels["artifact"].astype(bool))
    good_idx = np.flatnonzero(good_mask.to_numpy())

    picks = mne.pick_types(epochs.info, eeg=True, emg=True)
    X_all = epochs.get_data(picks=picks)             # (N_all, C, T)
    X = X_all[good_idx]
    y = labels.loc[good_idx, "stage"].map(STAGE_TO_INT).to_numpy()

    # -------- 6) Meta --------
    meta = {
        "edf_path": edf_path,
        "scoring_path": scoring_path,
        "sfreq": sf,
        "epoch_len_sec": elen,
        "samples_per_epoch": spe,
        "line_freq": line_freq,
        "line_notch_width": line_notch_width if line_notch_width else None,
        "eeg_band": eeg_band,
        "emg_band": emg_band,
        "resample_hz": resample_hz,
        "ch_names": [epochs.ch_names[i] for i in picks],
        "n_epochs_total": int(len(epochs)),
        "n_epochs_kept": int(len(X)),
        "kept_indices": good_idx,
        "stage_counts_all": labels["stage"].value_counts().to_dict(),
        "stage_counts_kept": labels.loc[good_idx, "stage"].value_counts().to_dict(),
        "unknown_count": int((labels["stage"] == "Unknown").sum()),
        "artifact_count": int(labels["artifact"].sum()),
        "mapping": STAGE_TO_INT,
        "consensus_rule": consensus_rule,
    }
    return X, y, meta


def collate_results(results):
    """
    results: list of dicts with keys: 'epochs' (X_i), 'labels' (y_i), 'meta', 'animal_id', 'cohort'
    Returns:
        X  : (N, C, T) float32
        y  : (N,) int64
        groups : (N,) subject IDs (for GroupKFold)
        info : dict with channel names, sfreq, samples_per_epoch, cohorts per epoch, stage names
    """
    if not results:
        raise ValueError("results is empty")

    # Use the first recording as reference
    ref = results[0]["meta"]
    ref_ch = list(ref["ch_names"])
    ref_spe = int(ref["samples_per_epoch"])
    ref_sf  = float(ref["sfreq"])

    X_list, y_list, groups, cohorts = [], [], [], []
    for i, r in enumerate(results):
        X_i = r["epochs"]           # (n_i, C_i, T_i)
        y_i = r["labels"]           # (n_i,)
        m   = r["meta"]

        # Reorder channels to reference order if needed
        ch_i = list(m["ch_names"])
        if ch_i != ref_ch:
            try:
                reorder = [ch_i.index(ch) for ch in ref_ch]  # where each ref channel sits in this recording
            except ValueError as e:
                raise ValueError(f"Channel sets differ: expected {ref_ch}, got {ch_i}") from e
            X_i = X_i[:, reorder, :]

        # Append
        X_list.append(X_i.astype(np.float32, copy=False))
        y_list.append(np.asarray(y_i, dtype=np.int64))
        groups.extend([r.get("animal_id", f"rec{i}")] * len(y_i))
        cohorts.extend([r.get("cohort", None)] * len(y_i))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.asarray(groups)
    cohorts = np.asarray(cohorts, dtype=object)

    # (Optional) stage names per epoch using the first meta’s mapping
    inv_stage = {v: k for k, v in results[0]["meta"]["mapping"].items()}
    stage_names = np.array([inv_stage[int(lbl)] for lbl in y], dtype=object)

    info = {
        "ch_names": ref_ch,
        "sfreq": ref_sf,
        "samples_per_epoch": ref_spe,
        "groups": groups,          # subject IDs aligned to X/y
        "cohorts": cohorts,        # cohort per epoch
        "stage_names": stage_names # string labels aligned to y
    }
    return X, y, info