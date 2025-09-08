
from pathlib import Path
import pandas as pd

def build_manifest_simple(root: Path) -> pd.DataFrame:
    """Simplest manifest builder.

    Assumptions:
      - Dataset layout:
          root/
            CohortA/
              recordings/*.edf
              scorings/*.csv
            CohortB/...
      - Recording and scoring filenames share the same stem (e.g., A1.edf ↔ A1.csv).
      - No metadata reading; just paths.

    Returns a DataFrame with:
      cohort, animal_id, edf_path, scoring_path, has_scoring
    """
    rows = []
    root = Path(root)
    for cohort_dir in sorted(root.iterdir()):
        if not cohort_dir.is_dir():
            continue
        if not cohort_dir.name.lower().startswith("cohort"):
            continue
        rec_dir = cohort_dir / "recordings"
        sco_dir = cohort_dir / "scorings"
        if not rec_dir.exists():
            continue
        for edf_path in sorted(rec_dir.glob("*.edf")):
            stem = edf_path.stem
            scoring_path = (sco_dir / f"{stem}.csv") if sco_dir.exists() else None
            if scoring_path and not scoring_path.exists():
                scoring_path = None
            rows.append({
                "cohort": cohort_dir.name,
                "animal_id": stem,
                "edf_path": str(edf_path),
                "scoring_path": str(scoring_path) if scoring_path else None,
                "has_scoring": bool(scoring_path)
            })
    return pd.DataFrame(rows)

def save_manifest(df: pd.DataFrame, out_csv: Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)