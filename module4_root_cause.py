"""Module 4 — explain the first alarm with transparent sensor contribution scores."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from module1_data import FEATURES, ROOT, write_json
from module2_baseline import load_baseline


INPUT_DIR = ROOT / "outputs" / "03_kalman"
OUTPUT_DIR = ROOT / "outputs" / "04_root_cause"


def main() -> None:
    events = pd.read_csv(INPUT_DIR / "fault_duration_results.csv")
    traces = pd.read_csv(INPUT_DIR / "fault_duration_traces.csv")
    baseline = load_baseline()
    rows = []
    all_scores = []

    for _, event in events.iterrows():
        event_id = int(event["event_id"])
        event_trace = traces[traces["event_id"] == event_id]
        alarm_second = int(event["alarm_delay_after_injection_s"])
        # The synthetic test harness knows the injected rise starts at t=0. Use
        # the immediately preceding ten seconds as the local healthy reference.
        before = event_trace[(event_trace["relative_seconds"] >= -10) & (event_trace["relative_seconds"] <= -1)]
        current = event_trace[event_trace["relative_seconds"] == alarm_second].iloc[0]
        event_scores = []

        for feature_index, feature in enumerate(FEATURES):
            reference = float(before[feature].median())
            raw_change = float(current[feature]) - reference
            standardized_change = raw_change / float(baseline["scaler"].scale_[feature_index])
            contribution = abs(standardized_change)
            score = {
                "event_id": event_id,
                "feature": feature,
                "raw_change_at_alarm": raw_change,
                "standardized_change_at_alarm": standardized_change,
                "absolute_standardized_change": contribution,
                "direction": "high" if raw_change >= 0 else "low",
            }
            event_scores.append(score)
            all_scores.append(score)

        event_scores.sort(key=lambda item: item["absolute_standardized_change"], reverse=True)
        top = event_scores[0]
        rows.append({
            "event_id": event_id,
            "fault_duration_s": int(event["duration_seconds"]),
            "diagnosed_variable": top["feature"],
            "diagnosed_direction": top["direction"],
            "top_absolute_standardized_change": top["absolute_standardized_change"],
            "synthetic_ground_truth_variable": event["variable"],
            "diagnosis_correct": top["feature"] == event["variable"],
            "alarm_second": alarm_second,
        })

    diagnosis = pd.DataFrame(rows)
    scores = pd.DataFrame(all_scores)
    diagnosis.to_csv(OUTPUT_DIR / "root_cause_results.csv", index=False)
    scores.to_csv(OUTPUT_DIR / "root_cause_scores.csv", index=False)

    selected = diagnosis[diagnosis["fault_duration_s"].isin([10, 30, 100, 300, 600])]
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(len(selected), 1, figsize=(12, 14))
    for axis, (_, diagnosis_row) in zip(axes, selected.iterrows()):
        event_scores = scores[scores["event_id"] == diagnosis_row["event_id"]].sort_values(
            "absolute_standardized_change", ascending=False
        ).head(5)
        axis.bar(
            event_scores["feature"],
            event_scores["absolute_standardized_change"],
            color=["#0065A8"] + ["#8DBCD4"] * 4,
        )
        axis.set_title(
            f"{int(diagnosis_row['fault_duration_s'])} s case: diagnosed {diagnosis_row['diagnosed_variable']} | synthetic truth {diagnosis_row['synthetic_ground_truth_variable']}"
        )
        axis.set_ylabel("Absolute standardized change")
    axes[-1].set_xlabel("Sensor variable")
    figure.suptitle("Module 4 — Root cause at the first red alarm point", fontsize=15)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "root_cause_at_alarm.png", dpi=190, bbox_inches="tight")
    plt.close(figure)

    accuracy = float(diagnosis["diagnosis_correct"].mean())
    summary = {
        "events": len(diagnosis),
        "correct_against_synthetic_label": int(diagnosis["diagnosis_correct"].sum()),
        "synthetic_root_cause_accuracy": accuracy,
        "score_definition": "|sensor at alarm - median of ten seconds before rise| / training SD",
        "reference_window": "ten seconds immediately before the known synthetic rise; a deployed version needs an onset detector",
        "field_fault_labels_used": False,
    }
    write_json(OUTPUT_DIR / "root_cause_summary.json", summary)
    checks = {
        "status": "PASS" if accuracy >= 0.80 else "FAIL",
        "accuracy_at_least_80_percent": bool(accuracy >= 0.80),
        "one_diagnosis_per_event": bool(len(diagnosis) == len(events)),
        "all_scores_finite": bool(np.isfinite(scores["absolute_standardized_change"]).all()),
    }
    write_json(OUTPUT_DIR / "module_check.json", checks)
    if checks["status"] != "PASS":
        raise AssertionError(checks)

    print("MODULE 4 PASS - root cause evaluated at the first alarm")
    print(diagnosis.to_string(index=False))


if __name__ == "__main__":
    main()
