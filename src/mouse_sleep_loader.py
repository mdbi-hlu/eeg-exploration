
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

import pandas as pd
import mne

@dataclass
class RecordingMeta:
    n_channels: Optional[int] = None
    channel_names: Optional[List[str]] = None
    sample_rate: Optional[float] = None
    start_time_iso: Optional[str] = None
    duration_sec: Optional[float] = None
    edf_format: Optional[str] = None
    annotations_preview: Optional[List[Dict[str, Any]]] = None

    def to_json(self) -> str:
        return json.dumps({
            "n_channels": self.n_channels,
            "channel_names": self.channel_names,
            "sample_rate": self.sample_rate,
            "start_time_iso": self.start_time_iso,
            "duration_sec": self.duration_sec,
            "edf_format": self.edf_format,
            "annotations_preview": self.annotations_preview,
        })

def read_edf_metadata(edf_path: Path) -> RecordingMeta:
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, stim_channel=None, verbose=False)
    info = raw.info
    ch_names = info.get('ch_names', [])
    sfreq = float(info['sfreq']) if isinstance(info['sfreq'], (int, float)) else float(info['sfreq'])
    onset = info.get('meas_date')
    start_iso = onset.isoformat() if onset is not None else None
    # Duration
    dur = float(raw._last_time) if hasattr(raw, "_last_time") else float(raw.times[-1]) if len(raw.times) else None
    # Annotations preview (first few)
    ann_prev = None
    if raw.annotations is not None and len(raw.annotations) > 0:
        ann_prev = []
        for i in range(min(5, len(raw.annotations))):
            a = raw.annotations[i]
            ann_prev.append({"onset": float(a["onset"]), "duration": float(a["duration"]), "desc": str(a["description"])})
    return RecordingMeta(
        n_channels=len(ch_names),
        channel_names=ch_names,
        sample_rate=sfreq,
        start_time_iso=start_iso,
        duration_sec=dur,
        edf_format="edf (mne)",
        annotations_preview=ann_prev
    )

def build_manifest(root: Path) -> pd.DataFrame:
    """Walk the dataset and create a manifest DataFrame.

    Layout expected:
        root/
          CohortA/
            recordings/*.edf
            scorings/*.csv
          CohortB/...
    Columns:
        cohort, animal_id, edf_path, scoring_path, meta_json, n_channels, duration_sec, start_time_iso, sample_rate
    """
    rows = []
    for cohort_dir in sorted(root.glob("Cohort*/")):
        cohort = cohort_dir.name
        rec_dir = cohort_dir / "recordings"
        sco_dir = cohort_dir / "scorings"
        if not rec_dir.exists():
            continue
        for edf_path in sorted(rec_dir.glob("*.edf")):
            stem = edf_path.stem  # e.g., 'A1'
            print(stem)
            animal_id = stem
            scoring_path = None
            if sco_dir.exists():
                candidate = sco_dir / f"{stem}.csv"
                if candidate.exists():
                    scoring_path = candidate
                else:
                    for c in sco_dir.glob("*.csv"):
                        if c.stem.lower() == stem.lower():
                            scoring_path = c
                            break
            meta = read_edf_metadata(edf_path)
            rows.append({
                "cohort": cohort,
                "animal_id": animal_id,
                "edf_path": str(edf_path),
                "scoring_path": str(scoring_path) if scoring_path else None,
                "meta_json": meta.to_json(),
                "n_channels": meta.n_channels,
                "duration_sec": meta.duration_sec,
                "start_time_iso": meta.start_time_iso,
                "sample_rate": meta.sample_rate
            })
    df = pd.DataFrame(rows)
    return df

def save_manifest(df: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

def load_recording(edf_path: Path, preload: bool = True) -> mne.io.BaseRaw:
    """Load EDF as an mne.io.Raw object."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=preload, stim_channel=None, verbose=False)
    return raw

# ---------- Scoring utilities ----------

def _normalize_scoring_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize scoring CSV to columns: onset_sec, duration_sec, stage.
    Heuristics cover common variants (onset/duration, start/end, epoch+stage).
    Adjust here if your schema differs.
    """
    cols = {c.lower(): c for c in df.columns}
    # Direct mapping
    if 'onset' in cols and 'duration' in cols:
        out = pd.DataFrame({
            'onset_sec': pd.to_numeric(df[cols['onset']], errors='coerce'),
            'duration_sec': pd.to_numeric(df[cols['duration']], errors='coerce'),
            'stage': df.get(cols.get('stage', ''), df.iloc[:, -1])
        })
        return out.dropna(subset=['onset_sec','duration_sec']).reset_index(drop=True)
    # Start/End times
    if 'start_time' in cols and 'end_time' in cols:
        st = pd.to_numeric(df[cols['start_time']], errors='coerce')
        en = pd.to_numeric(df[cols['end_time']], errors='coerce')
        out = pd.DataFrame({
            'onset_sec': st,
            'duration_sec': en - st,
            'stage': df.get(cols.get('stage', ''), df.iloc[:, -1])
        })
        return out.dropna(subset=['onset_sec','duration_sec']).reset_index(drop=True)
    # Epoch-based
    if 'epoch' in cols:
        epoch_len = 20.0
        if 'time' in cols:
            s = pd.to_numeric(df[cols['time']], errors='coerce').dropna().diff().mode()
            if len(s):
                epoch_len = float(s.iloc[0])
        out = pd.DataFrame({
            'onset_sec': (pd.to_numeric(df[cols['epoch']], errors='coerce') - 1) * epoch_len,
            'duration_sec': epoch_len,
            'stage': df.get(cols.get('stage', ''), df.iloc[:, -1])
        })
        return out.dropna(subset=['onset_sec']).reset_index(drop=True)
    # Fallback: find two numeric columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) >= 2:
        out = pd.DataFrame({
            'onset_sec': pd.to_numeric(df[num_cols[0]], errors='coerce'),
            'duration_sec': pd.to_numeric(df[num_cols[1]], errors='coerce'),
            'stage': df.iloc[:, -1]
        })
        return out.dropna(subset=['onset_sec','duration_sec']).reset_index(drop=True)
    raise ValueError("Could not normalize scoring CSV columns. Please adapt _normalize_scoring_df to your schema.")

def load_scoring(scoring_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(scoring_csv)
    df_norm = _normalize_scoring_df(df)
    # Stage normalization mapping
    mapping = {
        'W': 'Wake', 'WAKE': 'Wake', '0':'Wake', 0:'Wake',
        'N1':'NREM', 'N2':'NREM', 'N3':'NREM', '1':'NREM', '2':'NREM', '3':'NREM',
        'SWS':'NREM', 'NREM':'NREM',
        'R':'REM', 'REM':'REM', '5':'REM'
    }
    def _map_stage(x):
        if pd.isna(x):
            return x
        s = str(x).strip().upper()
        return mapping.get(s, x)
    df_norm['stage_norm'] = df_norm['stage'].map(_map_stage)
    return df_norm

def align_scoring_to_signal(scoring_df: pd.DataFrame, signal_duration_sec: Optional[float]) -> pd.DataFrame:
    df = scoring_df.copy()
    df['end_sec'] = df['onset_sec'] + df['duration_sec']
    if signal_duration_sec is not None:
        df = df[df['onset_sec'] < signal_duration_sec].copy()
        df.loc[df['end_sec'] > signal_duration_sec, 'end_sec'] = signal_duration_sec
        df['duration_sec'] = df['end_sec'] - df['onset_sec']
        df = df[df['duration_sec'] > 0].reset_index(drop=True)
    return df

# ---------- MNE integration helpers ----------

def scoring_to_annotations(scoring_df: pd.DataFrame) -> mne.Annotations:
    """Convert scoring rows to mne.Annotations (description = stage_norm if present else stage)."""
    desc = scoring_df['stage_norm'] if 'stage_norm' in scoring_df.columns else scoring_df['stage']
    ann = mne.Annotations(onset=scoring_df['onset_sec'].to_numpy(),
                          duration=scoring_df['duration_sec'].to_numpy(),
                          description=desc.astype(str).to_numpy())
    return ann

def attach_annotations(raw: mne.io.BaseRaw, scoring_df: pd.DataFrame) -> mne.io.BaseRaw:
    """Attach scoring annotations to an mne.Raw object (non-destructive; returns the same Raw)."""
    ann = scoring_to_annotations(scoring_df)
    if raw.annotations is None or len(raw.annotations) == 0:
        raw.set_annotations(ann)
    else:
        raw.set_annotations(raw.annotations + ann)
    return raw

def events_from_scoring(scoring_df: pd.DataFrame, event_id: Optional[Dict[str, int]] = None) -> Tuple[Any, Dict[str,int]]:
    """Create (events, event_id) suitable for mne.Epochs from scoring.

    events shape: (n, 3) with columns: [sample, 0, event_code]
    Default event_id: {'Wake': 1, 'NREM': 2, 'REM': 3}
    """
    import numpy as np
    if event_id is None:
        event_id = {'Wake': 1, 'NREM': 2, 'REM': 3}
    labels = scoring_df['stage_norm'] if 'stage_norm' in scoring_df.columns else scoring_df['stage'].astype(str)
    # Sample index will need sfreq (provided later when making Epochs). Here we keep seconds for now and scale outside.
    events_sec = scoring_df['onset_sec'].to_numpy()
    codes = np.array([event_id.get(str(l), 0) for l in labels], dtype=int)
    events = np.c_[events_sec, np.zeros_like(codes), codes]  # seconds in first col for now
    return events, event_id

def seconds_to_samples(events_sec: Any, sfreq: float) -> Any:
    """Utility to convert events array with seconds in first column to samples."""
    import numpy as np
    ev = np.array(events_sec, dtype=float).copy()
    ev[:, 0] = np.round(ev[:, 0] * sfreq).astype(int)
    return ev.astype(int)

def example_usage():
    # Example (not executed):
    # from pathlib import Path
    # root = Path('/path/to/dataset/root')
    # df = build_manifest(root)
    # raw = load_recording(Path(df.iloc[0]['edf_path']), preload=False)
    # scoring = load_scoring(Path(df.iloc[0]['scoring_path']))
    # scoring = align_scoring_to_signal(scoring, df.iloc[0]['duration_sec'])
    # raw = attach_annotations(raw, scoring)
    # # Create events for Epochs:
    # events_sec, event_id = events_from_scoring(scoring)
    # events = seconds_to_samples(events_sec, raw.info['sfreq'])
    # # Now you can build Epochs with a fixed length (e.g., 20s) if desired.
    pass
