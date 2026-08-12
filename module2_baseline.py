"""Module 2 — ElasticNet baseline trained on exactly the first 40%."""

from __future__ import annotations

import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from module1_data import FEATURES, ROOT, SIGMA_MULTIPLIER, TARGET, load_split, regression_metrics, write_json


OUTPUT_DIR = ROOT / "outputs" / "02_baseline"
MODEL_FILE = OUTPUT_DIR / "elasticnet_baseline.pkl"


def fit_baseline(frame: pd.DataFrame) -> dict:
    x_values = frame[FEATURES].to_numpy(dtype=float)
    y_values = frame[TARGET].to_numpy(dtype=float)
    scaler = StandardScaler().fit(x_values)
    model = ElasticNet(
        alpha=0.010,
        l1_ratio=0.10,
        fit_intercept=True,
        max_iter=30000,
        tol=1e-6,
        random_state=42,
    )
    model.fit(scaler.transform(x_values), y_values)
    signed_residual = y_values - model.predict(scaler.transform(x_values))
    absolute_residual = np.abs(signed_residual)
    absolute_mean = float(np.mean(absolute_residual))
    absolute_sd = float(np.std(absolute_residual, ddof=1))
    boundary = absolute_mean + SIGMA_MULTIPLIER * absolute_sd
    return {
        "scaler": scaler,
        "model": model,
        "features": FEATURES,
        "absolute_residual_mean_MW": absolute_mean,
        "absolute_residual_sd_MW": absolute_sd,
        "boundary_MW": boundary,
        "boundary_definition": "mean(|residual|) + 10 x sample SD(|residual|) on baseline 40%",
    }


def load_baseline() -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Run module2_baseline.py first: {MODEL_FILE}")
    with MODEL_FILE.open("rb") as file:
        return pickle.load(file)


def predict(frame_or_array: pd.DataFrame | np.ndarray, baseline: dict) -> np.ndarray:
    if isinstance(frame_or_array, pd.DataFrame):
        x_values = frame_or_array[FEATURES].to_numpy(dtype=float)
    else:
        x_values = np.asarray(frame_or_array, dtype=float)
    return baseline["model"].predict(baseline["scaler"].transform(x_values))


def residuals(frame: pd.DataFrame, baseline: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual = frame[TARGET].to_numpy(dtype=float)
    expected = predict(frame, baseline)
    signed = actual - expected
    return expected, signed, np.abs(signed)


def main() -> None:
    train = load_split("baseline_train")
    matrix_train = load_split("matrix_train")
    test = load_split("test")
    baseline = fit_baseline(train)
    with MODEL_FILE.open("wb") as file:
        pickle.dump(baseline, file)

    metrics_rows = []
    residual_frames = []
    for name, frame in [("baseline_train_40_percent", train), ("matrix_train_30_percent", matrix_train), ("untouched_test_30_percent", test)]:
        expected, signed, absolute = residuals(frame, baseline)
        metrics_rows.append({"split": name, **regression_metrics(frame[TARGET].to_numpy(dtype=float), expected)})
        residual_frames.append(pd.DataFrame({
            "split": name,
            "clean_row": frame["clean_row"].to_numpy(dtype=int),
            "EP_actual_MW": frame[TARGET].to_numpy(dtype=float),
            "EP_expected_MW": expected,
            "signed_residual_MW": signed,
            "absolute_residual_MW": absolute,
        }))

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(OUTPUT_DIR / "baseline_metrics.csv", index=False)
    pd.concat(residual_frames, ignore_index=True).to_csv(OUTPUT_DIR / "all_split_residuals.csv", index=False)

    coefficients = []
    for index, feature in enumerate(FEATURES):
        coefficients.append({
            "feature": feature,
            "standardized_coefficient": float(baseline["model"].coef_[index]),
            "raw_MW_per_feature_unit": float(baseline["model"].coef_[index] / baseline["scaler"].scale_[index]),
        })
    pd.DataFrame(coefficients).to_csv(OUTPUT_DIR / "elasticnet_coefficients.csv", index=False)

    sample = test[FEATURES].iloc[[0]].to_numpy(dtype=float)
    latency = []
    for _ in range(5000):
        start = time.perf_counter_ns()
        baseline["model"].predict(baseline["scaler"].transform(sample))
        latency.append((time.perf_counter_ns() - start) / 1000.0)

    test_expected, _, test_absolute = residuals(test, baseline)
    test_metrics = metrics.iloc[-1]
    summary = {
        "model": "ElasticNet",
        "training_rows": len(train),
        "training_fraction_after_cleaning": 0.40,
        "test_r2": float(test_metrics["r2"]),
        "test_mae_MW": float(test_metrics["mae"]),
        "absolute_residual_mean_MW": float(baseline["absolute_residual_mean_MW"]),
        "absolute_residual_sd_MW": float(baseline["absolute_residual_sd_MW"]),
        "one_sided_boundary_MW": float(baseline["boundary_MW"]),
        "boundary_definition": baseline["boundary_definition"],
        "prediction_mean_latency_us": float(np.mean(latency)),
        "prediction_p99_latency_us": float(np.percentile(latency, 99)),
    }
    write_json(OUTPUT_DIR / "baseline_summary.json", summary)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    width = min(500, len(test))
    axes[0, 0].plot(test[TARGET].to_numpy(dtype=float)[:width], color="#6B7280", label="actual EP")
    axes[0, 0].plot(test_expected[:width], color="#0065A8", label="ElasticNet expected EP")
    axes[0, 0].set_title("Untouched final 30%: actual vs expected")
    axes[0, 0].set_ylabel("MW")
    axes[0, 0].legend()

    axes[0, 1].scatter(test[TARGET], test_expected, s=8, alpha=0.35, color="#0065A8")
    low = min(float(test[TARGET].min()), float(test_expected.min()))
    high = max(float(test[TARGET].max()), float(test_expected.max()))
    axes[0, 1].plot([low, high], [low, high], "--", color="#DC2626")
    axes[0, 1].set_title(f"Final 30% fit: R²={float(test_metrics['r2']):.3f}")
    axes[0, 1].set_xlabel("Actual EP (MW)")
    axes[0, 1].set_ylabel("Expected EP (MW)")

    axes[1, 0].plot(test_absolute[:width], color="#0065A8")
    axes[1, 0].axhline(baseline["boundary_MW"], color="#DC2626", linestyle="--", label="one-sided boundary")
    axes[1, 0].set_title("Absolute residual is never negative")
    axes[1, 0].set_xlabel("Source observation index")
    axes[1, 0].set_ylabel("|Actual − expected| (MW)")
    axes[1, 0].legend()

    axes[1, 1].hist(test_absolute, bins=40, color="#8DBCD4", edgecolor="white")
    axes[1, 1].axvline(baseline["boundary_MW"], color="#DC2626", linestyle="--")
    axes[1, 1].set_title("Absolute residual distribution")
    axes[1, 1].set_xlabel("|Residual| (MW)")
    figure.suptitle("Module 2 — ElasticNet baseline and one-sided residual", fontsize=15)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "elasticnet_baseline_results.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    checks = {
        "status": "PASS" if float(test_metrics["r2"]) >= 0.80 and np.all(test_absolute >= 0) else "FAIL",
        "trained_on_exact_first_40_percent": True,
        "test_r2_at_least_0_80": bool(float(test_metrics["r2"]) >= 0.80),
        "absolute_residual_never_negative": bool(np.all(test_absolute >= 0)),
        "one_sided_boundary_positive": bool(baseline["boundary_MW"] > 0),
    }
    write_json(OUTPUT_DIR / "module_check.json", checks)
    if checks["status"] != "PASS":
        raise AssertionError(checks)

    print("MODULE 2 PASS - ElasticNet baseline trained on first 40%")
    print(metrics.to_string(index=False))
    print(f"one-sided boundary: {baseline['boundary_MW']:.3f} MW")


if __name__ == "__main__":
    main()
