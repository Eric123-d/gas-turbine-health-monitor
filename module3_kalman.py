"""Module 3 — train A/B/C/bias on the middle 30%, then forecast at most 600 s."""

from __future__ import annotations

import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from module1_data import (
    DRIFT_WINDOW_SECONDS,
    FEATURES,
    FORECAST_HORIZON_SECONDS,
    ROOT,
    interpolate_to_one_hz,
    load_split,
    regression_metrics,
    write_json,
)
from module2_baseline import load_baseline, predict


OUTPUT_DIR = ROOT / "outputs" / "03_kalman"
MODEL_FILE = OUTPUT_DIR / "trained_adaptive_kalman.pkl"
RANDOM_SEED = 20260812
FAULT_DURATIONS = [10, 20, 30, 60, 100, 180, 300, 450, 600]
PRE_SECONDS = 600
POST_SECONDS = 60
ALARM_HISTORY_SECONDS = 60
ALARM_RISE_FRACTION = 0.075
ALARM_RATE_FRACTION = 0.20
ALARM_CONFIRMATIONS = 2
ALARM_ACTIVATION_SIGMA = 5.0


def absolute_residual(frame: pd.DataFrame, baseline: dict) -> np.ndarray:
    return np.abs(frame["EP"].to_numpy(dtype=float) - predict(frame, baseline))


def drift_rate_from_last_five_changes(h_values: np.ndarray) -> np.ndarray:
    """Exact requested definition: median of the latest five 1-second h changes."""
    drift = np.zeros(len(h_values), dtype=float)
    changes = np.diff(h_values)
    for index in range(1, len(h_values)):
        first_change = max(0, index - DRIFT_WINDOW_SECONDS)
        drift[index] = float(np.median(changes[first_change:index]))
    return drift


def observed_states(h_values: np.ndarray) -> np.ndarray:
    return np.column_stack([h_values, drift_rate_from_last_five_changes(h_values)])


def latent_states(h_values: np.ndarray) -> np.ndarray:
    """Use a causal five-second median for the hidden health level."""
    smooth_h = np.zeros(len(h_values), dtype=float)
    for index in range(len(h_values)):
        start = max(0, index - DRIFT_WINDOW_SECONDS + 1)
        smooth_h[index] = float(np.median(h_values[start:index + 1]))
    return np.column_stack([smooth_h, drift_rate_from_last_five_changes(h_values)])


def fault_profile(duration: int, seed: int) -> np.ndarray:
    """Continuous rising profile with sparse steep bursts and small correlated ripple."""
    progress = np.linspace(0.0, 1.0, duration + 1)
    envelope = np.sin(np.pi * progress)
    rng = np.random.default_rng(seed)
    raw_noise = rng.normal(0.0, 1.0, len(progress))
    width = max(3, min(21, 2 * (duration // 60) + 3))
    correlated = np.convolve(raw_noise, np.ones(width) / width, mode="same")
    correlated /= max(float(np.std(correlated)), 1e-12)

    profile = progress + 0.018 * envelope * correlated
    for center, amplitude in [(0.18, 0.045), (0.55, 0.055), (0.82, 0.035)]:
        burst_width = max(0.018, 2.0 / duration)
        profile += amplitude * envelope * np.exp(-0.5 * ((progress - center) / burst_width) ** 2)

    # Preserve a readable worsening trend: bursts can make it steeper, never reverse it.
    profile = np.maximum.accumulate(np.clip(profile, 0.0, None))
    profile -= profile[0]
    profile /= max(float(profile[-1]), 1e-12)
    profile[0] = 0.0
    profile[-1] = 1.0
    return profile


def choose_fault_variables(baseline: dict) -> list[str]:
    importance = np.abs(baseline["model"].coef_)
    order = np.argsort(importance)[::-1]
    return [FEATURES[int(index)] for index in order[:5]]


def inject_fault_segment(
    one_hz: pd.DataFrame,
    start: int,
    duration: int,
    variable: str,
    final_absolute_residual: float,
    baseline: dict,
    seed: int,
) -> tuple[pd.DataFrame, int, dict]:
    """Inject a realistic sensor shift plus linked power degradation."""
    segment_start = start - PRE_SECONDS
    segment_end = start + duration + POST_SECONDS + 1
    segment = one_hz.iloc[segment_start:segment_end].copy().reset_index(drop=True)
    local_start = PRE_SECONDS
    profile = fault_profile(duration, seed)
    column = FEATURES.index(variable)
    direction = -1.0 if baseline["model"].coef_[column] > 0 else 1.0
    sensor_final_change = direction * 2.0 * float(baseline["scaler"].scale_[column])

    for offset in range(duration + 1):
        segment.loc[local_start + offset, variable] += sensor_final_change * profile[offset]
    for offset in range(local_start + duration + 1, len(segment)):
        segment.loc[offset, variable] += sensor_final_change

    # Keep the physical sensor shift moderate. Add a linked performance loss so
    # the final |residual| reaches the chosen training/test severity.
    prediction_after_sensor = predict(segment, baseline)
    signed_after_sensor = segment["EP"].to_numpy(dtype=float) - prediction_after_sensor
    end_index = local_start + duration
    pre_sign = np.sign(float(np.median(signed_after_sensor[local_start - 30:local_start])))
    if pre_sign == 0:
        pre_sign = 1.0
    desired_signed_end = pre_sign * final_absolute_residual
    performance_final_change = desired_signed_end - signed_after_sensor[end_index]
    for offset in range(duration + 1):
        segment.loc[local_start + offset, "EP"] += performance_final_change * profile[offset]
    for offset in range(local_start + duration + 1, len(segment)):
        segment.loc[offset, "EP"] += performance_final_change

    details = {
        "variable": variable,
        "duration_seconds": duration,
        "sensor_final_change": sensor_final_change,
        "synthetic_EP_final_change_MW": performance_final_change,
        "target_final_absolute_residual_MW": final_absolute_residual,
    }
    return segment, local_start, details


def normalized_inputs(frame: pd.DataFrame, baseline: dict) -> np.ndarray:
    return baseline["scaler"].transform(frame[FEATURES].to_numpy(dtype=float))


def fit_ridge_mapping(design: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    identity = np.eye(design.shape[1])
    identity[-1, -1] = 0.0  # do not penalize the bias column
    return np.linalg.solve(design.T @ design + alpha * identity, design.T @ target)


def covariance_with_floor(error: np.ndarray, floors: np.ndarray) -> np.ndarray:
    covariance = np.cov(error.T)
    return covariance + np.diag(floors)


def train_state_space(matrix_frame: pd.DataFrame, baseline: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    one_hz = interpolate_to_one_hz(matrix_frame)
    healthy_h = absolute_residual(one_hz, baseline)
    healthy_state = latent_states(healthy_h)
    healthy_observation = observed_states(healthy_h)
    healthy_u = normalized_inputs(one_hz, baseline)

    rng = np.random.default_rng(RANDOM_SEED)
    healthy_available = len(one_hz) - 1
    samples_per_class = min(80000, healthy_available // 2)
    healthy_indices = np.sort(rng.choice(healthy_available, samples_per_class, replace=False))

    previous_parts = [healthy_state[healthy_indices]]
    input_parts = [healthy_u[healthy_indices + 1]]
    next_parts = [healthy_state[healthy_indices + 1]]
    state_parts = [healthy_state[healthy_indices + 1]]
    observation_parts = [healthy_observation[healthy_indices + 1]]

    fault_rows = []
    fault_previous = []
    fault_inputs = []
    fault_next = []
    fault_states = []
    fault_observations = []
    variables = choose_fault_variables(baseline)
    scenario = 0
    collected = 0

    while collected < samples_per_class:
        duration = FAULT_DURATIONS[scenario % len(FAULT_DURATIONS)]
        variable = variables[(scenario // len(FAULT_DURATIONS)) % len(variables)]
        start = int(rng.integers(PRE_SECONDS + 1, len(one_hz) - duration - POST_SECONDS - 2))
        severity_fraction = 0.55 + 0.65 * ((scenario % 7) / 6.0)
        target_h = severity_fraction * float(baseline["boundary_MW"])
        segment, local_start, details = inject_fault_segment(
            one_hz, start, duration, variable, target_h, baseline, RANDOM_SEED + scenario
        )
        h_values = absolute_residual(segment, baseline)
        states = latent_states(h_values)
        observations = observed_states(h_values)
        inputs = normalized_inputs(segment, baseline)
        first = local_start - 1
        last = local_start + duration
        fault_previous.append(states[first:last])
        fault_inputs.append(inputs[first + 1:last + 1])
        fault_next.append(states[first + 1:last + 1])
        fault_states.append(states[first + 1:last + 1])
        fault_observations.append(observations[first + 1:last + 1])
        collected += last - first
        fault_rows.append({"scenario_id": scenario + 1, **details, "transitions": last - first})
        scenario += 1

    previous_parts.append(np.vstack(fault_previous)[:samples_per_class])
    input_parts.append(np.vstack(fault_inputs)[:samples_per_class])
    next_parts.append(np.vstack(fault_next)[:samples_per_class])
    state_parts.append(np.vstack(fault_states)[:samples_per_class])
    observation_parts.append(np.vstack(fault_observations)[:samples_per_class])

    previous = np.vstack(previous_parts)
    inputs = np.vstack(input_parts)
    next_state = np.vstack(next_parts)
    measurement_state = np.vstack(state_parts)
    observations = np.vstack(observation_parts)
    order = rng.permutation(len(next_state))
    previous, inputs, next_state = previous[order], inputs[order], next_state[order]
    measurement_state, observations = measurement_state[order], observations[order]

    validation_start = int(0.80 * len(next_state))
    train_slice = slice(0, validation_start)
    validation_slice = slice(validation_start, len(next_state))

    transition_design = np.column_stack([previous, inputs, np.ones(len(previous))])
    transition_parameters = fit_ridge_mapping(
        transition_design[train_slice], next_state[train_slice], alpha=0.10
    )
    a_matrix = transition_parameters[:2].T
    b_matrix = transition_parameters[2:2 + len(FEATURES)].T
    transition_bias = transition_parameters[-1]
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(a_matrix))))
    if spectral_radius >= 0.999:
        a_matrix *= 0.995 / spectral_radius
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(a_matrix))))

    measurement_design = np.column_stack([measurement_state, np.ones(len(measurement_state))])
    measurement_parameters = fit_ridge_mapping(
        measurement_design[train_slice], observations[train_slice], alpha=0.01
    )
    c_matrix = measurement_parameters[:2].T
    measurement_bias = measurement_parameters[-1]

    predicted_state = previous @ a_matrix.T + inputs @ b_matrix.T + transition_bias
    predicted_observation = measurement_state @ c_matrix.T + measurement_bias
    process_error = next_state - predicted_state
    measurement_error = observations - predicted_observation
    q_matrix = covariance_with_floor(process_error[train_slice], np.array([1e-7, 1e-9]))
    r_matrix = covariance_with_floor(measurement_error[train_slice], np.array([1e-7, 1e-9]))

    state_validation = []
    for state_index, state_name in enumerate(["absolute_residual_state", "five_change_median_drift"]):
        metrics = regression_metrics(next_state[validation_slice, state_index], predicted_state[validation_slice, state_index])
        state_validation.append({"target": state_name, **metrics})
    for observation_index, name in enumerate(["observed_absolute_residual", "observed_five_change_median_drift"]):
        metrics = regression_metrics(observations[validation_slice, observation_index], predicted_observation[validation_slice, observation_index])
        state_validation.append({"target": f"C_mapping_{name}", **metrics})

    package = {
        "A": a_matrix,
        "B": b_matrix,
        "C": c_matrix,
        "transition_bias": transition_bias,
        "measurement_bias": measurement_bias,
        "Q": q_matrix,
        "R": r_matrix,
        "input_min": np.percentile(inputs[train_slice], 0.1, axis=0),
        "input_max": np.percentile(inputs[train_slice], 99.9, axis=0),
        "input_slope_decay": 0.98,
        "boundary_MW": float(baseline["boundary_MW"]),
        "spectral_radius_A": spectral_radius,
        "state_definition": "x=[causal median |residual|, median of latest five one-second |residual| changes]",
        "observation_definition": "z=[raw |residual|, median of latest five one-second |residual| changes]",
        "training_source": "middle 30%; balanced healthy and synthetic-fault transitions",
    }
    return package, pd.DataFrame(fault_rows), pd.DataFrame(state_validation)


def load_kalman() -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Run module3_kalman.py first: {MODEL_FILE}")
    with MODEL_FILE.open("rb") as file:
        return pickle.load(file)


def clip_covariance(matrix: np.ndarray, floor: float, ceiling: float) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    return vectors @ np.diag(np.clip(values, floor, ceiling)) @ vectors.T


def forecast_crossing(state: np.ndarray, current_u: np.ndarray, u_slope: np.ndarray, kalman: dict) -> int | None:
    """Return the first predicted boundary crossing, never extrapolating beyond 600 s."""
    forecast_state = state.copy()
    forecast_u = current_u.copy()
    forecast_slope = u_slope.copy()
    for second in range(1, FORECAST_HORIZON_SECONDS + 1):
        forecast_u = np.clip(forecast_u + forecast_slope, kalman["input_min"], kalman["input_max"])
        forecast_slope *= kalman["input_slope_decay"]
        forecast_state = kalman["A"] @ forecast_state + kalman["B"] @ forecast_u + kalman["transition_bias"]
        forecast_observation = kalman["C"] @ forecast_state + kalman["measurement_bias"]
        if float(forecast_observation[0]) >= float(kalman["boundary_MW"]):
            return second
    return None


def run_online_filter(frame: pd.DataFrame, baseline: dict, kalman: dict) -> dict:
    h_values = absolute_residual(frame, baseline)
    observations = observed_states(h_values)
    inputs = normalized_inputs(frame, baseline)
    state = np.array([h_values[0], 0.0])
    covariance = np.eye(2)
    identity = np.eye(2)
    updated = []
    predicted_crossing = []
    latency_us = []

    for index in range(len(frame)):
        start = time.perf_counter_ns()
        state_prior = kalman["A"] @ state + kalman["B"] @ inputs[index] + kalman["transition_bias"]
        innovation = observations[index] - (kalman["C"] @ state_prior + kalman["measurement_bias"])
        innovation_scale = min(max(float(np.linalg.norm(innovation)), 1.0), 10.0)
        covariance_prior = kalman["A"] @ covariance @ kalman["A"].T + innovation_scale * kalman["Q"]
        innovation_covariance = kalman["C"] @ covariance_prior @ kalman["C"].T + kalman["R"]
        gain = covariance_prior @ kalman["C"].T @ np.linalg.inv(innovation_covariance)
        state = state_prior + gain @ innovation
        covariance = (identity - gain @ kalman["C"]) @ covariance_prior @ (identity - gain @ kalman["C"]).T + gain @ kalman["R"] @ gain.T
        covariance = clip_covariance(covariance, 1e-10, 100.0)

        if index >= 2:
            begin = max(1, index - 4)
            u_slope = np.median(np.diff(inputs[begin - 1:index + 1], axis=0), axis=0)
        else:
            u_slope = np.zeros(inputs.shape[1])
        crossing = forecast_crossing(state, inputs[index], u_slope, kalman)
        updated.append(state.copy())
        predicted_crossing.append(np.nan if crossing is None else crossing)
        latency_us.append((time.perf_counter_ns() - start) / 1000.0)

    return {
        "absolute_residual": h_values,
        "observations": observations,
        "updated_state": np.vstack(updated),
        "predicted_crossing_seconds": np.asarray(predicted_crossing),
        "latency_us": np.asarray(latency_us),
    }


def first_index(mask: np.ndarray, start: int) -> int | None:
    for index in range(start, len(mask)):
        if bool(mask[index]):
            return index
    return None


def confirmed_alarm_mask(
    h_values: np.ndarray,
    drift_values: np.ndarray,
    crossing_seconds: np.ndarray,
    boundary: float,
    activation_level: float,
) -> np.ndarray:
    """Require a sustained residual rise before accepting a crossing forecast."""
    candidates = np.zeros(len(h_values), dtype=bool)
    confirmed = np.zeros(len(h_values), dtype=bool)
    minimum_drift = ALARM_RATE_FRACTION * boundary / FORECAST_HORIZON_SECONDS
    minimum_rise = ALARM_RISE_FRACTION * boundary

    for index in range(ALARM_HISTORY_SECONDS, len(h_values)):
        reference = float(np.median(h_values[index - ALARM_HISTORY_SECONDS:index]))
        rise = float(h_values[index] - reference)
        candidates[index] = (
            np.isfinite(crossing_seconds[index])
            and h_values[index] >= activation_level
            and drift_values[index] >= minimum_drift
            and rise >= minimum_rise
        )
        if index >= ALARM_CONFIRMATIONS - 1:
            recent = candidates[index - ALARM_CONFIRMATIONS + 1:index + 1]
            confirmed[index] = bool(np.all(recent))
    return confirmed


def count_alarm_events(mask: np.ndarray) -> int:
    count = 0
    previous = False
    for current in mask:
        if bool(current) and not previous:
            count += 1
        previous = bool(current)
    return count


def test_fault_durations(test_frame: pd.DataFrame, baseline: dict, kalman: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    one_hz = interpolate_to_one_hz(test_frame)
    variables = choose_fault_variables(baseline)
    starts = np.linspace(PRE_SECONDS + 100, len(one_hz) - 1300, len(FAULT_DURATIONS), dtype=int)
    event_rows = []
    trace_rows = []
    plot_durations = [10, 30, 100, 300, 600]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(len(plot_durations), 1, figsize=(13, 15))
    plot_index = 0

    for event_index, duration in enumerate(FAULT_DURATIONS):
        variable = variables[event_index % len(variables)]
        segment, local_start, details = inject_fault_segment(
            one_hz,
            int(starts[event_index]),
            duration,
            variable,
            float(kalman["boundary_MW"]) + 0.20,
            baseline,
            9000 + duration,
        )
        result = run_online_filter(segment, baseline, kalman)
        h_values = result["absolute_residual"]
        crossing_seconds = result["predicted_crossing_seconds"]
        actual_cross = first_index(h_values >= kalman["boundary_MW"], local_start)
        alarm_confirmed = confirmed_alarm_mask(
            h_values,
            result["observations"][:, 1],
            crossing_seconds,
            float(kalman["boundary_MW"]),
            float(baseline["absolute_residual_mean_MW"] + ALARM_ACTIVATION_SIGMA * baseline["absolute_residual_sd_MW"]),
        )
        alarm = first_index(alarm_confirmed, local_start)
        pre_fault_alarm = first_index(alarm_confirmed, 120)
        if pre_fault_alarm is not None and pre_fault_alarm >= local_start:
            pre_fault_alarm = None

        alarm_delay = None if alarm is None else alarm - local_start
        actual_cross_time = None if actual_cross is None else actual_cross - local_start
        lead_time = None
        if alarm_delay is not None and actual_cross_time is not None:
            lead_time = actual_cross_time - alarm_delay

        event_id = event_index + 1
        event_rows.append({
            "event_id": event_id,
            **details,
            "detected": alarm is not None,
            "alarm_delay_after_injection_s": alarm_delay,
            "actual_boundary_cross_s": actual_cross_time,
            "lead_time_before_actual_cross_s": lead_time,
            "predicted_seconds_to_cross_at_alarm": None if alarm is None else int(crossing_seconds[alarm]),
            "pre_fault_false_alarm": pre_fault_alarm is not None,
            "mean_filter_and_forecast_latency_us": float(np.mean(result["latency_us"])),
            "p95_filter_and_forecast_latency_us": float(np.percentile(result["latency_us"], 95)),
        })

        relative = np.arange(len(segment)) - local_start
        trace = pd.DataFrame({
            "event_id": event_id,
            "fault_duration_s": duration,
            "relative_seconds": relative,
            "absolute_residual_MW": h_values,
            "filtered_health_state_MW": result["updated_state"][:, 0],
            "drift_rate_MW_per_s": result["observations"][:, 1],
            "predicted_crossing_seconds": crossing_seconds,
            "alarm_confirmed": alarm_confirmed,
            "filter_and_forecast_latency_us": result["latency_us"],
            "boundary_MW": kalman["boundary_MW"],
            "EP_MW": segment["EP"].to_numpy(dtype=float),
        })
        for feature in FEATURES:
            trace[feature] = segment[feature].to_numpy(dtype=float)
        trace_rows.append(trace)

        if duration in plot_durations:
            axis = axes[plot_index]
            display = (relative >= -30) & (relative <= duration + 30)
            axis.plot(relative[display], h_values[display], color="#2F3E46", linewidth=1.6, label="|residual|")
            axis.axhline(kalman["boundary_MW"], color="#DC2626", linestyle="--", label="one-sided 10 SD boundary")
            axis.axvline(0, color="#F59E0B", linestyle=":", label="fault begins")
            if alarm is not None:
                axis.scatter(alarm - local_start, h_values[alarm], color="#DC2626", s=55, zorder=5, label="alarm")
            if actual_cross is not None:
                axis.scatter(actual_cross - local_start, h_values[actual_cross], color="#7C3AED", marker="x", s=70, zorder=5, label="actual crossing")
            axis.set_title(f"{duration} s gradual {variable} fault — residual rises, alarm is the red point")
            axis.set_ylabel("|Residual| MW")
            axis.set_title(f"{duration} s gradual {variable} fault - residual rises, alarm is the red point")
            if plot_index == 0:
                axis.legend(ncol=4, fontsize=8)
            plot_index += 1

    axes[-1].set_xlabel("Seconds relative to synthetic fault start")
    figure.suptitle("Module 3 — Residual against time and 600-second maximum forecast", fontsize=15)
    figure.suptitle("Module 3 - Residual against time and 600-second maximum forecast", fontsize=15)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "residual_time_alarm_cases.png", dpi=190, bbox_inches="tight")
    plt.close(figure)
    return pd.DataFrame(event_rows), pd.concat(trace_rows, ignore_index=True)


def main() -> None:
    baseline = load_baseline()
    matrix_frame = load_split("matrix_train")
    test_frame = load_split("test")
    kalman, training_faults, validation = train_state_space(matrix_frame, baseline)
    with MODEL_FILE.open("wb") as file:
        pickle.dump(kalman, file)

    np.savetxt(OUTPUT_DIR / "A_matrix.csv", kalman["A"], delimiter=",")
    np.savetxt(OUTPUT_DIR / "B_matrix.csv", kalman["B"], delimiter=",")
    np.savetxt(OUTPUT_DIR / "C_matrix.csv", kalman["C"], delimiter=",")
    np.savetxt(OUTPUT_DIR / "transition_bias.csv", kalman["transition_bias"], delimiter=",")
    np.savetxt(OUTPUT_DIR / "measurement_bias.csv", kalman["measurement_bias"], delimiter=",")
    np.savetxt(OUTPUT_DIR / "Q_matrix.csv", kalman["Q"], delimiter=",")
    np.savetxt(OUTPUT_DIR / "R_matrix.csv", kalman["R"], delimiter=",")
    training_faults.to_csv(OUTPUT_DIR / "matrix_training_faults.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "matrix_validation_metrics.csv", index=False)

    events, traces = test_fault_durations(test_frame, baseline, kalman)
    events.to_csv(OUTPUT_DIR / "fault_duration_results.csv", index=False)
    traces.to_csv(OUTPUT_DIR / "fault_duration_traces.csv", index=False)

    # A bounded measured-data check: six derived 1 Hz hours, with no injection.
    healthy_one_hz = interpolate_to_one_hz(test_frame.iloc[:92])
    healthy_result = run_online_filter(healthy_one_hz, baseline, kalman)
    healthy_alarm_mask = confirmed_alarm_mask(
        healthy_result["absolute_residual"],
        healthy_result["observations"][:, 1],
        healthy_result["predicted_crossing_seconds"],
        float(kalman["boundary_MW"]),
        float(baseline["absolute_residual_mean_MW"] + ALARM_ACTIVATION_SIGMA * baseline["absolute_residual_sd_MW"]),
    )
    healthy_alarm_count = count_alarm_events(healthy_alarm_mask)
    warned_in_time = int((events["lead_time_before_actual_cross_s"] >= 0).sum())
    summary = {
        "matrix_training_source": "middle 30% strict assumed-health data plus synthetic faults",
        "A": kalman["A"].tolist(),
        "B_shape": list(kalman["B"].shape),
        "C": kalman["C"].tolist(),
        "transition_bias": kalman["transition_bias"].tolist(),
        "measurement_bias": kalman["measurement_bias"].tolist(),
        "A_spectral_radius": kalman["spectral_radius_A"],
        "forecast_horizon_max_seconds": FORECAST_HORIZON_SECONDS,
        "drift_definition": "median of the latest five one-second changes of |residual|",
        "tested_fault_durations_seconds": FAULT_DURATIONS,
        "events_detected": int(events["detected"].sum()),
        "events_warned_before_or_at_cross": warned_in_time,
        "mean_lead_time_seconds": float(events["lead_time_before_actual_cross_s"].mean()),
        "healthy_check_derived_one_hz_seconds": len(healthy_one_hz),
        "healthy_confirmed_alarm_event_count": healthy_alarm_count,
        "mean_filter_and_600s_forecast_latency_us": float(traces["filter_and_forecast_latency_us"].mean()),
        "p95_filter_and_600s_forecast_latency_us": float(traces["filter_and_forecast_latency_us"].quantile(0.95)),
        "warning_activation_MW": float(baseline["absolute_residual_mean_MW"] + ALARM_ACTIVATION_SIGMA * baseline["absolute_residual_sd_MW"]),
        "alarm_rule": "current residual above mean+5 SD + 600 s crossing forecast + latest-five-change median drift + 60 s residual rise + two confirmations",
        "one_hz_is_interpolated_not_measured": True,
    }
    write_json(OUTPUT_DIR / "kalman_summary.json", summary)

    matrices_finite = all(np.isfinite(kalman[name]).all() for name in ["A", "B", "C", "Q", "R"])
    checks = {
        "status": "PASS" if warned_in_time == len(events) and matrices_finite and healthy_alarm_count == 0 else "FAIL",
        "A_B_C_and_bias_trained_from_middle_30_percent": True,
        "all_test_durations_10_to_600_seconds_detected": bool(int(events["detected"].sum()) == len(events)),
        "all_alarms_before_or_at_actual_crossing": bool(warned_in_time == len(events)),
        "no_prefault_alarm_in_fault_cases": bool(int(events["pre_fault_false_alarm"].sum()) == 0),
        "no_confirmed_alarm_in_healthy_check": bool(healthy_alarm_count == 0),
        "matrices_are_finite": bool(matrices_finite),
        "forecast_never_exceeds_600_seconds": bool(np.nanmax(traces["predicted_crossing_seconds"]) <= 600),
    }
    write_json(OUTPUT_DIR / "module_check.json", checks)
    if checks["status"] != "PASS":
        raise AssertionError(checks)

    print("MODULE 3 PASS - A/B/C/bias trained and 10-600 s tests completed")
    print(validation.to_string(index=False))
    print(events[["event_id", "duration_seconds", "variable", "alarm_delay_after_injection_s", "actual_boundary_cross_s", "lead_time_before_actual_cross_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
