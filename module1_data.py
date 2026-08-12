"""Module 1 — strict assumed-health cleaning, 40/30/30 split, shared functions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "GTPP-DATASET.csv"
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs" / "01_data_split"

FEATURES = ["AT", "AP", "DF", "CH", "GP", "CPR", "CPD", "TTXM"]
TARGET = "EP"
SOURCE_INTERVAL_SECONDS = 240
ONLINE_INTERVAL_SECONDS = 1
FORECAST_HORIZON_SECONDS = 600
DRIFT_WINDOW_SECONDS = 5
SIGMA_MULTIPLIER = 10.0

# Wide engineering plausibility limits. They catch corrupt values, not faults.
PHYSICAL_RANGES = {
    "AT": (-20.0, 60.0),
    "AP": (25.0, 32.0),
    "DF": (48.0, 52.0),
    "CH": (0.0, 0.10),
    "GP": (4.0, 10.0),
    "CPR": (6.0, 18.0),
    "CPD": (6.0, 16.0),
    "TTXM": (400.0, 700.0),
    "EP": (50.0, 150.0),
}


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["01_data_split", "02_baseline", "03_kalman", "04_root_cause", "05_adjustment"]:
        (ROOT / "outputs" / name).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def robust_scale(values: np.ndarray) -> float:
    """Median absolute deviation on a standard-deviation scale."""
    median = float(np.median(values))
    return max(1.4826 * float(np.median(np.abs(values - median))), 1e-12)


def source_sha256() -> str:
    digest = hashlib.sha256()
    with RAW_FILE.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_data() -> pd.DataFrame:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing dataset: {RAW_FILE}")
    frame = pd.read_csv(RAW_FILE)
    expected = FEATURES + [TARGET]
    if frame.columns.tolist() != expected:
        raise ValueError(f"Expected {expected}, found {frame.columns.tolist()}")
    frame = frame.apply(pd.to_numeric, errors="raise")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Raw data contains missing or non-finite values.")
    return frame


def isolated_spike_mask(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    """Find isolated five-point Hampel spikes while preserving real ramps."""
    any_spike = np.zeros(len(frame), dtype=bool)
    counts: dict[str, int] = {}

    for column in FEATURES + [TARGET]:
        values = frame[column].to_numpy(dtype=float)
        differences = np.diff(values)
        difference_scale = robust_scale(differences)
        column_mask = np.zeros(len(values), dtype=bool)

        for index in range(2, len(values) - 2):
            neighbors = np.concatenate([values[index - 2:index], values[index + 1:index + 3]])
            local_median = float(np.median(neighbors))
            local_scale = max(robust_scale(neighbors), 0.10 * difference_scale)
            neighbors_rejoin = abs(values[index - 1] - values[index + 1]) <= 4.0 * difference_scale
            if abs(values[index] - local_median) > 8.0 * local_scale and neighbors_rejoin:
                column_mask[index] = True

        any_spike |= column_mask
        counts[column] = int(column_mask.sum())
    return any_spike, counts


def provisional_residual_screen(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    """Remove rows incompatible with a robust model learned only from the first 40%."""
    split = int(0.40 * len(frame))
    x_values = frame[FEATURES].to_numpy(dtype=float)
    y_values = frame[TARGET].to_numpy(dtype=float)
    scaler = StandardScaler().fit(x_values[:split])
    model = ElasticNet(alpha=0.002, l1_ratio=0.10, max_iter=30000, random_state=42)
    model.fit(scaler.transform(x_values[:split]), y_values[:split])
    signed_residual = y_values - model.predict(scaler.transform(x_values))
    center = float(np.median(signed_residual[:split]))
    scale = robust_scale(signed_residual[:split])
    keep = np.abs(signed_residual - center) <= 6.0 * scale
    return keep, {"residual_center_MW": center, "residual_robust_scale_MW": scale, "cutoff_MW": 6.0 * scale}


def clean_assumed_healthy_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply auditable, deterministic health-reference screens."""
    audit_rows = []
    keep = np.ones(len(raw), dtype=bool)

    physical = np.ones(len(raw), dtype=bool)
    for column, (lower, upper) in PHYSICAL_RANGES.items():
        physical &= raw[column].between(lower, upper).to_numpy()
    audit_rows.append({"stage": "physical ranges and finite values", "removed": int((~physical).sum())})
    keep &= physical

    duplicates = raw.duplicated().to_numpy()
    audit_rows.append({"stage": "exact duplicate rows", "removed": int((keep & duplicates).sum())})
    keep &= ~duplicates

    spike_mask, spike_counts = isolated_spike_mask(raw)
    audit_rows.append({"stage": "isolated five-point Hampel spikes", "removed": int((keep & spike_mask).sum())})
    keep &= ~spike_mask

    first_pass = raw.loc[keep].copy()
    first_pass.insert(0, "source_row", first_pass.index.astype(int))
    first_pass.reset_index(drop=True, inplace=True)
    residual_keep, residual_details = provisional_residual_screen(first_pass)
    audit_rows.append({"stage": "six-MAD provisional baseline residual screen", "removed": int((~residual_keep).sum())})

    cleaned = first_pass.loc[residual_keep].reset_index(drop=True)
    cleaned.insert(1, "clean_row", np.arange(len(cleaned), dtype=int))
    details = {
        "spikes_by_variable": spike_counts,
        "provisional_residual_screen": residual_details,
    }
    return cleaned, pd.DataFrame(audit_rows), details


def split_40_30_30(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    first_end = int(0.40 * len(frame))
    second_end = int(0.70 * len(frame))
    return {
        "baseline_train": frame.iloc[:first_end].copy(),
        "matrix_train": frame.iloc[first_end:second_end].copy(),
        "test": frame.iloc[second_end:].copy(),
    }


def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run module1_data.py first: {path}")
    return pd.read_csv(path)


def interpolate_to_one_hz(frame: pd.DataFrame) -> pd.DataFrame:
    """Linearly derive a 1 Hz simulation timeline from 240-second public rows."""
    columns = FEATURES + [TARGET]
    source = frame[columns].to_numpy(dtype=float)
    rows: list[np.ndarray] = []
    seconds: list[int] = []
    current_second = 0

    for index in range(len(source) - 1):
        start = source[index]
        end = source[index + 1]
        for offset in range(SOURCE_INTERVAL_SECONDS):
            fraction = offset / SOURCE_INTERVAL_SECONDS
            rows.append(start + fraction * (end - start))
            seconds.append(current_second)
            current_second += 1
    rows.append(source[-1])
    seconds.append(current_second)

    result = pd.DataFrame(np.vstack(rows), columns=columns)
    result.insert(0, "time_seconds", seconds)
    return result


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "r2": 1.0 - float(np.sum(error * error)) / denominator,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
    }


def main() -> None:
    ensure_directories()
    raw = load_raw_data()
    cleaned, audit, details = clean_assumed_healthy_data(raw)
    splits = split_40_30_30(cleaned)

    cleaned.to_csv(DATA_DIR / "strict_assumed_healthy.csv", index=False)
    audit.to_csv(OUTPUT_DIR / "cleaning_audit.csv", index=False)
    summary_rows = []
    for name, split in splits.items():
        split.to_csv(DATA_DIR / f"{name}.csv", index=False)
        summary_rows.append({
            "split": name,
            "rows": len(split),
            "percent_of_cleaned": 100.0 * len(split) / len(cleaned),
            "first_clean_row": int(split["clean_row"].iloc[0]),
            "last_clean_row": int(split["clean_row"].iloc[-1]),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)

    removed = len(raw) - len(cleaned)
    checks = {
        "status": "PASS",
        "raw_rows": len(raw),
        "strict_assumed_healthy_rows": len(cleaned),
        "removed_rows": removed,
        "removed_percent": 100.0 * removed / len(raw),
        "missing_or_nonfinite_after_cleaning": int(cleaned.isna().sum().sum()),
        "exact_split_ratio": "40% / 30% / 30% after cleaning (integer floor at boundaries)",
        "health_labels_available": False,
        "health_claim": "strictly screened assumed-health reference; not label-confirmed healthy",
        "source_interval_seconds_from_paper": SOURCE_INTERVAL_SECONDS,
        "online_interval_seconds": ONLINE_INTERVAL_SECONDS,
        "one_hz_points_are_interpolated": True,
        "source_sha256": source_sha256(),
        **details,
    }
    write_json(OUTPUT_DIR / "module_check.json", checks)

    print("MODULE 1 PASS - strict assumed-health cleaning and 40/30/30 split")
    print(audit.to_string(index=False))
    print(summary.to_string(index=False))
    print("Important: no health label exists; 1 Hz points are interpolated from 240 s rows.")


if __name__ == "__main__":
    main()
