import numpy as np
import pandas as pd
import mne
from typing import Optional, Tuple


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

def preprocess_recording_and_label(
    edf_path: str,
    scoring_path: str,
    *,
    epoch_len: float = 4.0,
    line_freq: float = 50.0,                # set 60.0 if needed
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
    #raw.notch_filter(freqs=[line_freq], picks="all", verbose=False)
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