"""Module 1: clean the single data file and make an ordered 40/30/30 split."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data.csv"
FEATURES = ["AT", "AP", "DF", "CH", "GP", "CPR", "CPD", "TTXM"]
TARGET = "EP"
SAMPLE_INTERVAL_SECONDS = 240
FORECAST_STEPS = 600

PHYSICAL_RANGES = {
    "AT": (-20.0, 60.0), "AP": (25.0, 32.0), "DF": (48.0, 52.0),
    "CH": (0.0, 0.10), "GP": (4.0, 10.0), "CPR": (6.0, 18.0),
    "CPD": (6.0, 16.0), "TTXM": (400.0, 700.0), "EP": (50.0, 150.0),
}


def robust_sd(values):
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    return max(1.4826 * float(np.median(np.abs(values - median))), 1e-12)


def isolated_spikes(frame):
    bad = np.zeros(len(frame), dtype=bool)
    for column in FEATURES + [TARGET]:
        values = frame[column].to_numpy(dtype=float)
        difference_sd = robust_sd(np.diff(values))
        for index in range(2, len(values) - 2):
            neighbors = np.r_[values[index - 2:index], values[index + 1:index + 3]]
            local_median = float(np.median(neighbors))
            local_sd = max(robust_sd(neighbors), 0.10 * difference_sd)
            rejoined = abs(values[index - 1] - values[index + 1]) <= 4.0 * difference_sd
            if rejoined and abs(values[index] - local_median) > 8.0 * local_sd:
                bad[index] = True
    return bad


def load_and_split():
    raw = pd.read_csv(DATA_FILE)
    if raw.columns.tolist() != FEATURES + [TARGET]:
        raise ValueError("data.csv columns do not match the documented gas-turbine schema")
    raw = raw.apply(pd.to_numeric, errors="raise")
    finite = np.isfinite(raw.to_numpy(dtype=float)).all(axis=1)
    physical = np.ones(len(raw), dtype=bool)
    for column, limits in PHYSICAL_RANGES.items():
        physical &= raw[column].between(*limits).to_numpy()
    duplicate = raw.duplicated().to_numpy()
    spike = isolated_spikes(raw)
    first_clean = raw.loc[finite & physical & ~duplicate & ~spike].reset_index(drop=True)

    # A provisional first-40% ElasticNet removes rows incompatible with the
    # assumed-healthy power relationship. No fault is injected in this module.
    split = int(0.40 * len(first_clean))
    scaler = StandardScaler().fit(first_clean[FEATURES].iloc[:split])
    model = ElasticNet(alpha=0.002, l1_ratio=0.10, max_iter=30000, random_state=42)
    model.fit(scaler.transform(first_clean[FEATURES].iloc[:split]), first_clean[TARGET].iloc[:split])
    residual = first_clean[TARGET].to_numpy() - model.predict(scaler.transform(first_clean[FEATURES]))
    center = float(np.median(residual[:split]))
    keep = np.abs(residual - center) <= 6.0 * robust_sd(residual[:split])
    clean = first_clean.loc[keep].reset_index(drop=True)

    first_end = int(0.40 * len(clean))
    second_end = int(0.70 * len(clean))
    parts = clean.iloc[:first_end], clean.iloc[first_end:second_end], clean.iloc[second_end:]
    removed = len(raw) - len(clean)
    print(f"Module 1 PASS | healthy-reference rows={len(clean)} | removed={removed} | split={list(map(len, parts))}")
    return clean, parts[0].copy(), parts[1].copy(), parts[2].copy()


if __name__ == "__main__":
    load_and_split()
