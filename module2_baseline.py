"""Module 2: ElasticNet healthy baseline and one-sided absolute residual."""

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from module1_data import FEATURES, TARGET


def predict(frame, baseline):
    values = baseline["scaler"].transform(frame[FEATURES])
    return baseline["model"].predict(values)


def absolute_residual(frame, baseline):
    return np.abs(frame[TARGET].to_numpy(dtype=float) - predict(frame, baseline))


def r2(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    return 1.0 - float(np.sum((actual - predicted) ** 2)) / float(np.sum((actual - actual.mean()) ** 2))


def train_baseline(train, test):
    scaler = StandardScaler().fit(train[FEATURES])
    model = ElasticNet(alpha=0.01, l1_ratio=0.10, max_iter=30000, tol=1e-6, random_state=42)
    model.fit(scaler.transform(train[FEATURES]), train[TARGET])
    baseline = {"scaler": scaler, "model": model}
    train_h = absolute_residual(train, baseline)
    baseline["residual_mean"] = float(train_h.mean())
    baseline["residual_sd"] = float(train_h.std(ddof=1))
    baseline["boundary"] = baseline["residual_mean"] + 10.0 * baseline["residual_sd"]
    train_r2 = r2(train[TARGET], predict(train, baseline))
    test_r2 = r2(test[TARGET], predict(test, baseline))
    test_mae = float(absolute_residual(test, baseline).mean())
    if test_r2 < 0.80 or baseline["boundary"] <= 0:
        raise AssertionError("ElasticNet baseline validation failed")
    print(f"Module 2 PASS | train R2={train_r2:.4f} | test R2={test_r2:.4f} | MAE={test_mae:.3f} | boundary={baseline['boundary']:.3f}")
    return baseline
