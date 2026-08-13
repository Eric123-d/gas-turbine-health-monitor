"""Module 3: learn A/B/C/bias and forecast at most 600 source-data steps."""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

from module1_data import FEATURES, FORECAST_STEPS, ROOT, SAMPLE_INTERVAL_SECONDS
from module2_baseline import absolute_residual, predict


FAULT_STEPS = [10, 30, 100, 300, 600]


def drift(values):
    values = np.asarray(values, dtype=float)
    result = np.zeros(len(values))
    changes = np.diff(values)
    for index in range(1, len(values)):
        result[index] = float(np.median(changes[max(0, index - 5):index]))
    return result


def states(values):
    return np.column_stack([values, drift(values)])


def fault_profile(length, seed):
    rng = np.random.default_rng(seed)
    progress = np.linspace(0.0, 1.0, length + 1)
    ripple = np.cumsum(np.maximum(rng.normal(0.0, 0.002, len(progress)), -0.001))
    profile = progress + np.sin(np.pi * progress) * ripple
    for center, height in [(0.20, 0.025), (0.55, 0.035), (0.82, 0.025)]:
        profile += height * np.exp(-0.5 * ((progress - center) / 0.025) ** 2)
    profile = np.maximum.accumulate(profile)
    profile = (profile - profile[0]) / (profile[-1] - profile[0])
    return profile


def standardized_inputs(frame, baseline):
    return baseline["scaler"].transform(frame[FEATURES])


def train_kalman(matrix_frame, baseline):
    rng = np.random.default_rng(20260813)
    healthy_h = absolute_residual(matrix_frame, baseline)
    healthy_x = states(healthy_h)
    healthy_u = standardized_inputs(matrix_frame, baseline)
    previous, current_u, next_state = [], [], []
    fault_previous, fault_next = [], []

    for index in range(1, len(matrix_frame)):
        previous.append(healthy_x[index - 1]); current_u.append(healthy_u[index]); next_state.append(healthy_x[index])

    # Many ramp rates and sensor directions teach persistent fault transitions.
    for scenario in range(180):
        length = int(rng.choice(FAULT_STEPS))
        feature = scenario % len(FEATURES)
        base = int(rng.integers(0, len(matrix_frame)))
        profile = fault_profile(length, scenario)
        target_h = baseline["boundary"] * rng.uniform(0.8, 1.25)
        h = healthy_h[base] + profile * (target_h - healthy_h[base])
        x = states(h)
        persistent_x = states(np.linspace(healthy_h[base], target_h, length + 1))
        sensor_change = (-1.0 if baseline["model"].coef_[feature] > 0 else 1.0) * rng.uniform(1.0, 2.5)
        u = np.repeat(healthy_u[[base]], length + 1, axis=0)
        u[:, feature] += sensor_change * profile
        for index in range(1, len(h)):
            previous.append(x[index - 1]); current_u.append(u[index]); next_state.append(x[index])
            fault_previous.append(persistent_x[index - 1]); fault_next.append(persistent_x[index])

    previous = np.asarray(previous); current_u = np.asarray(current_u); next_state = np.asarray(next_state)
    fault_previous = np.asarray(fault_previous); fault_next = np.asarray(fault_next)
    state_design = np.column_stack([fault_previous, np.ones(len(fault_previous))])
    state_model = Ridge(alpha=1e-12, fit_intercept=False).fit(state_design, fault_next)
    a_matrix = state_model.coef_[:, :2]; transition_bias = state_model.coef_[:, -1]
    remaining = next_state - previous @ a_matrix.T - transition_bias
    input_model = Ridge(alpha=20.0, fit_intercept=False).fit(current_u, remaining)
    b_matrix = input_model.coef_

    # C and measurement bias are trained, even though this simple state is close to directly observed.
    measurement_design = np.column_stack([next_state, np.ones(len(next_state))])
    measurement = Ridge(alpha=1e-6, fit_intercept=False).fit(measurement_design, next_state)
    c_matrix = measurement.coef_[:, :2]
    measurement_bias = measurement.coef_[:, -1]
    predicted = previous @ a_matrix.T + current_u @ b_matrix.T + transition_bias
    process_error = next_state - predicted
    measurement_error = next_state - measurement.predict(measurement_design)
    q_matrix = np.cov(process_error.T) + np.diag([1e-6, 1e-8])
    r_matrix = np.cov(measurement_error.T) + np.diag([1e-5, 1e-7])

    package = {"A": a_matrix, "B": b_matrix, "C": c_matrix, "tb": transition_bias,
               "mb": measurement_bias, "Q": q_matrix, "R": r_matrix, "boundary": baseline["boundary"]}
    if not all(np.isfinite(package[key]).all() for key in ["A", "B", "C", "Q", "R"]):
        raise AssertionError("non-finite Kalman matrix")
    print(f"Module 3 matrices | A={np.round(a_matrix, 4).tolist()} | C={np.round(c_matrix, 4).tolist()}")
    return package


def forecast_crossing(state, current_u, u_slope, kalman):
    future_state = state.copy(); future_u = current_u.copy()
    for step in range(1, FORECAST_STEPS + 1):
        future_u = future_u + u_slope
        future_state = kalman["A"] @ future_state + kalman["B"] @ future_u + kalman["tb"]
        observed = kalman["C"] @ future_state + kalman["mb"]
        if observed[0] >= kalman["boundary"]:
            return step
    return None


def filter_and_forecast(frame, baseline, kalman, forecast_limit):
    h = absolute_residual(frame, baseline); z = states(h); u = standardized_inputs(frame, baseline)
    x = z[0].copy(); p = np.eye(2); identity = np.eye(2)
    filtered, crossings, latency = [], [], []
    for index in range(len(frame)):
        started = time.perf_counter_ns()
        prior_x = kalman["A"] @ x + kalman["B"] @ u[index] + kalman["tb"]
        prior_p = kalman["A"] @ p @ kalman["A"].T + kalman["Q"]
        innovation = z[index] - (kalman["C"] @ prior_x + kalman["mb"])
        innovation_p = kalman["C"] @ prior_p @ kalman["C"].T + kalman["R"]
        gain = prior_p @ kalman["C"].T @ np.linalg.inv(innovation_p)
        x = prior_x + gain @ innovation
        p = (identity - gain @ kalman["C"]) @ prior_p
        start = max(1, index - 4)
        slope = np.median(np.diff(u[start - 1:index + 1], axis=0), axis=0) if index else np.zeros(u.shape[1])
        crossing = forecast_crossing(x, u[index], slope, kalman) if index <= forecast_limit else None
        filtered.append(x.copy()); crossings.append(np.nan if crossing is None else crossing)
        latency.append((time.perf_counter_ns() - started) / 1e6)
    return h, z, np.asarray(filtered), np.asarray(crossings), np.asarray(latency)


def synthetic_case(test_frame, baseline, duration, feature, seed):
    base = test_frame.iloc[[100 + seed * 31]].copy()
    length = 12 + duration + 20
    frame = base.loc[base.index.repeat(length)].reset_index(drop=True).astype(float)
    start = 10; profile = fault_profile(duration, seed + 500)
    column = FEATURES.index(feature)
    direction = -1.0 if baseline["model"].coef_[column] > 0 else 1.0
    sensor_change = direction * 2.0 * baseline["scaler"].scale_[column]
    for offset in range(duration + 1):
        frame.loc[start + offset, feature] += sensor_change * profile[offset]
    frame.loc[start + duration + 1:, feature] += sensor_change
    after_sensor = frame["EP"].to_numpy() - predict(frame, baseline)
    sign = 1.0 if after_sensor[start] >= 0 else -1.0
    final_change = sign * (baseline["boundary"] + 0.25) - after_sensor[start + duration]
    for offset in range(duration + 1):
        frame.loc[start + offset, "EP"] += final_change * profile[offset]
    frame.loc[start + duration + 1:, "EP"] += final_change
    return frame, start


def test_and_plot(test_frame, baseline, kalman):
    variables = [FEATURES[i] for i in np.argsort(np.abs(baseline["model"].coef_))[::-1][:5]]
    events = []; plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(5, 1, figsize=(12, 14))
    for case, (duration, feature) in enumerate(zip(FAULT_STEPS, variables)):
        frame, start = synthetic_case(test_frame, baseline, duration, feature, case)
        h, z, filtered, crossings, latency = filter_and_forecast(frame, baseline, kalman, start + 5)
        alarm = None
        for index in range(start + 2, len(frame)):
            recent = np.diff(h[max(start, index - 2):index + 1])
            if len(recent) >= 2 and np.all(recent > 0) and np.isfinite(crossings[index]):
                alarm = index; break
        actual = next((i for i in range(start, len(h)) if h[i] >= baseline["boundary"]), None)
        if alarm is None or actual is None or alarm > actual or alarm - start > 5:
            raise AssertionError(f"late forecast alarm for {duration}-step case")
        events.append({"duration": duration, "feature": feature, "alarm_step": alarm - start,
                       "cross_step": actual - start, "forecast_at_alarm": int(crossings[alarm]),
                       "latency_ms": float(np.mean(latency)), "frame": frame, "start": start,
                       "alarm_index": alarm, "h": h})
        relative = np.arange(len(frame)) - start
        display = (relative >= -5) & (relative <= duration + 10)
        axes[case].plot(relative[display], h[display], color="#263238", label="|residual|")
        axes[case].axhline(baseline["boundary"], color="#DC2626", linestyle="--", label="one-sided 10 SD boundary")
        axes[case].axvline(0, color="#F59E0B", linestyle=":", label="fault starts")
        axes[case].scatter(alarm - start, h[alarm], color="#DC2626", s=55, zorder=4, label="forecast alarm")
        axes[case].scatter(actual - start, h[actual], color="#7C3AED", marker="x", s=65, zorder=4, label="actual crossing")
        seconds = duration * SAMPLE_INTERVAL_SECONDS
        axes[case].set_title(f"{duration} samples = {seconds:,} s | alarm after {alarm-start} samples")
        axes[case].set_ylabel("|Residual| MW")
        if case == 0: axes[case].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel(f"Samples from fault start (one sample = {SAMPLE_INTERVAL_SECONDS} s)")
    figure.suptitle(f"Gas turbine: direct forecast up to {FORECAST_STEPS} samples ({FORECAST_STEPS*SAMPLE_INTERVAL_SECONDS:,} s)")
    figure.tight_layout(); figure.savefig(ROOT / "residual_alarm.png", dpi=170, bbox_inches="tight"); plt.close(figure)
    print("Module 3 PASS | " + ", ".join(f"{e['duration']} steps: alarm {e['alarm_step']}" for e in events))
    return events
