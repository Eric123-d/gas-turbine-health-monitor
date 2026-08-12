"""Module 5 — operator-supervised adjustment and self-contained result report."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from module1_data import FEATURES, ROOT, write_json
from module2_baseline import load_baseline


KALMAN_DIR = ROOT / "outputs" / "03_kalman"
CAUSE_DIR = ROOT / "outputs" / "04_root_cause"
OUTPUT_DIR = ROOT / "outputs" / "05_adjustment"
REPORT_FILE = ROOT / "outputs" / "PROJECT_REPORT.html"

ACTION_MAP = {
    "AT": "check inlet-temperature sensing/cooling and derate if necessary",
    "AP": "verify inlet pressure sensing and inlet restriction",
    "DF": "hold load and verify grid-frequency / generator measurements",
    "CH": "verify humidity sensor and inlet-condition compensation",
    "GP": "stabilize gas supply and inspect regulator or fuel-gas valve",
    "CPR": "derate and inspect compressor condition or fouling",
    "CPD": "reduce load, verify IGV position and inspect compressor delivery",
    "TTXM": "reduce fuel/load and inspect combustion and exhaust sensing",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def predict_one(values: np.ndarray, baseline: dict) -> float:
    standardized = baseline["scaler"].transform(values.reshape(1, -1))
    return float(baseline["model"].predict(standardized)[0])


def image_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def html_table(frame: pd.DataFrame) -> str:
    clean = frame.copy()
    for column in clean.select_dtypes(include=["number"]).columns:
        clean[column] = clean[column].map(
            lambda value: "" if pd.isna(value) else (str(int(value)) if abs(value - round(value)) < 1e-9 else f"{value:.4f}")
        )
    return clean.to_html(index=False, border=0, classes="result-table", escape=True)


def build_report(proposals: pd.DataFrame, recovery: pd.DataFrame) -> None:
    cleaning = pd.read_csv(ROOT / "outputs" / "01_data_split" / "cleaning_audit.csv")
    split_summary = pd.read_csv(ROOT / "outputs" / "01_data_split" / "split_summary.csv")
    baseline_metrics = pd.read_csv(ROOT / "outputs" / "02_baseline" / "baseline_metrics.csv")
    baseline_summary = read_json(ROOT / "outputs" / "02_baseline" / "baseline_summary.json")
    kalman_summary = read_json(KALMAN_DIR / "kalman_summary.json")
    events = pd.read_csv(KALMAN_DIR / "fault_duration_results.csv")
    matrix_validation = pd.read_csv(KALMAN_DIR / "matrix_validation_metrics.csv")
    a_matrix = pd.read_csv(KALMAN_DIR / "A_matrix.csv", header=None)
    b_matrix = pd.read_csv(KALMAN_DIR / "B_matrix.csv", header=None)
    c_matrix = pd.read_csv(KALMAN_DIR / "C_matrix.csv", header=None)
    a_matrix.columns = ["previous residual state", "previous drift state"]
    b_matrix.columns = FEATURES
    c_matrix.columns = ["latent residual state", "latent drift state"]
    diagnoses = pd.read_csv(CAUSE_DIR / "root_cause_results.csv")
    checks = []
    for module, folder in [(1, "01_data_split"), (2, "02_baseline"), (3, "03_kalman"), (4, "04_root_cause"), (5, "05_adjustment")]:
        checks.append({"module": module, "status": read_json(ROOT / "outputs" / folder / "module_check.json")["status"]})

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gas Turbine Health Monitor — Five Module Report</title><style>
body{{font-family:Arial,sans-serif;margin:0;background:#F4F7F9;color:#17202A;line-height:1.5}}main{{max-width:1180px;margin:auto;padding:28px}}
header{{background:linear-gradient(120deg,#002B49,#0065A8);color:white;padding:34px;border-radius:14px}}h1{{margin:0}}h2{{color:#003B5C;margin-top:34px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:20px 0}}.card{{background:white;padding:16px;border-left:5px solid #00A1DE;border-radius:9px}}
.value{{font-size:23px;font-weight:bold;color:#003B5C}}.warning{{background:#FFF7E6;border-left:5px solid #F59E0B;padding:15px;border-radius:6px}}
.result-table{{border-collapse:collapse;width:100%;background:white;font-size:13px;display:block;overflow-x:auto}}.result-table th{{background:#003B5C;color:white;padding:8px}}
.result-table td{{border-bottom:1px solid #DCE3E8;padding:8px;white-space:nowrap}}img{{width:100%;background:white;border-radius:10px;margin:10px 0;box-shadow:0 2px 8px #0001}}
code{{background:#E8EEF2;padding:2px 5px;border-radius:4px}}footer{{margin-top:35px;color:#5F6B73;font-size:13px}}</style></head><body><main>
<header><h1>Gas Turbine Health Monitoring</h1><div>40% ElasticNet → |residual| → trained A/B/C/bias → adaptive Kalman → 600 s maximum forecast</div></header>
<div class="cards">
<div class="card">Final 30% baseline R²<div class="value">{baseline_summary['test_r2']:.3f}</div></div>
<div class="card">One-sided boundary<div class="value">{baseline_summary['one_sided_boundary_MW']:.3f} MW</div></div>
<div class="card">10–600 s events detected<div class="value">{kalman_summary['events_detected']}/9</div></div>
<div class="card">Mean warning lead<div class="value">{kalman_summary['mean_lead_time_seconds']:.1f} s</div></div>
<div class="card">Healthy confirmed alarms<div class="value">{kalman_summary['healthy_confirmed_alarm_event_count']}</div></div>
<div class="card">P95 filter + forecast latency<div class="value">{kalman_summary['p95_filter_and_600s_forecast_latency_us']/1000.0:.2f} ms</div></div>
</div>
<div class="warning"><b>Evidence boundary:</b> the public CSV has no health/fault labels and no absolute timestamps. Data are strictly screened but remain assumed-health, not label-confirmed healthy. The original interval is 240 s; all one-second Kalman points are explicitly interpolated/simulated. Synthetic fault accuracy is not field-fault accuracy. The boundary is statistical, not an OEM trip limit.</div>
<h2>1. Strict cleaning and 40/30/30 split</h2>{html_table(cleaning)}{html_table(split_summary)}
<h2>2. ElasticNet baseline and nonnegative residual</h2>{html_table(baseline_metrics)}<img src="{image_uri(ROOT / 'outputs' / '02_baseline' / 'elasticnet_baseline_results.png')}">
<h2>3. Trained A/B/C/bias and maximum 600-second forecast</h2>
<p><code>x(t)=A x(t-1)+B u(t)+bias</code>; <code>z(t)=C x(t)+measurement bias</code>. Drift is the median of the latest five one-second changes of |residual|.</p>
<h3>A matrix</h3>{html_table(a_matrix)}<h3>B matrix</h3>{html_table(b_matrix)}<h3>C matrix</h3>{html_table(c_matrix)}
{html_table(matrix_validation)}{html_table(events[['duration_seconds','variable','alarm_delay_after_injection_s','actual_boundary_cross_s','lead_time_before_actual_cross_s','predicted_seconds_to_cross_at_alarm','mean_filter_and_forecast_latency_us','p95_filter_and_forecast_latency_us']])}
<img src="{image_uri(KALMAN_DIR / 'residual_time_alarm_cases.png')}">
<h2>4. Root cause at the first alarm</h2>{html_table(diagnoses)}<img src="{image_uri(CAUSE_DIR / 'root_cause_at_alarm.png')}">
<h2>5. Supervised adjustment and return to reference</h2>{html_table(proposals)}<img src="{image_uri(OUTPUT_DIR / 'recovery_to_reference.png')}">
<h2>Automatic module checks</h2>{html_table(pd.DataFrame(checks))}
<h2>Reproduce</h2><p>Install <code>requirements.txt</code>, then run <code>./run_all.ps1</code>. Each module writes CSV/JSON evidence and fails loudly when its checks fail.</p>
<footer>SMART-GTPP Dataset DOI 10.17632/6sk3mhm7hb.1, CC BY 4.0. Educational prototype only; no controller commands are issued.</footer>
</main></body></html>"""
    REPORT_FILE.write_text(html, encoding="utf-8")


def main() -> None:
    diagnoses = pd.read_csv(CAUSE_DIR / "root_cause_results.csv")
    traces = pd.read_csv(KALMAN_DIR / "fault_duration_traces.csv")
    baseline = load_baseline()
    proposals = []
    recovery_rows = []

    selected_durations = [10, 30, 100, 300, 600]
    selected = diagnoses[diagnoses["fault_duration_s"].isin(selected_durations)]
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(len(selected), 1, figsize=(12, 14))
    plot_index = 0

    for _, diagnosis in diagnoses.iterrows():
        event_id = int(diagnosis["event_id"])
        duration = int(diagnosis["fault_duration_s"])
        variable = str(diagnosis["diagnosed_variable"])
        event_trace = traces[traces["event_id"] == event_id]
        reference_rows = event_trace[(event_trace["relative_seconds"] >= -60) & (event_trace["relative_seconds"] <= -5)]
        fault_end = event_trace[event_trace["relative_seconds"] == duration].iloc[0]
        reference_x = reference_rows[FEATURES].median().to_numpy(dtype=float)
        reference_ep = float(reference_rows["EP_MW"].median())
        current_x = fault_end[FEATURES].to_numpy(dtype=float)
        current_ep = float(fault_end["EP_MW"])
        event_recovery = []

        for step in range(11):
            fraction = step / 10.0
            smooth = fraction * fraction * (3.0 - 2.0 * fraction)
            adjusted_x = current_x + smooth * (reference_x - current_x)
            adjusted_ep = current_ep + smooth * (reference_ep - current_ep)
            expected_ep = predict_one(adjusted_x, baseline)
            h_value = abs(adjusted_ep - expected_ep)
            row = {
                "event_id": event_id,
                "elapsed_adjustment_seconds": step * 60,
                "adjustment_progress": smooth,
                "counterfactual_EP_MW": adjusted_ep,
                "counterfactual_expected_EP_MW": expected_ep,
                "counterfactual_absolute_residual_MW": h_value,
            }
            recovery_rows.append(row)
            event_recovery.append(row)

        end_h = event_recovery[-1]["counterfactual_absolute_residual_MW"]
        proposals.append({
            "event_id": event_id,
            "fault_duration_s": duration,
            "diagnosed_variable": variable,
            "operator_recommendation": ACTION_MAP[variable],
            "human_approval_required": True,
            "direct_controller_command": False,
            "counterfactual_final_absolute_residual_MW": end_h,
            "returned_below_one_sided_boundary": end_h < baseline["boundary_MW"],
        })

        if duration in selected_durations:
            axis = axes[plot_index]
            axis.plot(
                [row["elapsed_adjustment_seconds"] for row in event_recovery],
                [row["counterfactual_absolute_residual_MW"] for row in event_recovery],
                marker="o", color="#0065A8",
            )
            axis.axhline(baseline["boundary_MW"], color="#DC2626", linestyle="--", label="one-sided boundary")
            axis.set_title(f"{duration} s event: supervised return after diagnosing {variable}")
            axis.set_ylabel("|Residual| MW")
            if plot_index == 0:
                axis.legend()
            plot_index += 1

    proposal_frame = pd.DataFrame(proposals)
    recovery_frame = pd.DataFrame(recovery_rows)
    proposal_frame.to_csv(OUTPUT_DIR / "adjustment_proposals.csv", index=False)
    recovery_frame.to_csv(OUTPUT_DIR / "recovery_trace.csv", index=False)
    axes[-1].set_xlabel("Counterfactual adjustment time (s)")
    figure.suptitle("Module 5 — Return of nonnegative residual below the boundary", fontsize=15)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "recovery_to_reference.png", dpi=190, bbox_inches="tight")
    plt.close(figure)

    passed = int(proposal_frame["returned_below_one_sided_boundary"].sum())
    summary = {
        "events": len(proposal_frame),
        "events_returned_below_boundary": passed,
        "recovery_horizon_seconds": 600,
        "human_approval_required": True,
        "direct_controller_commands_issued": False,
    }
    write_json(OUTPUT_DIR / "adjustment_summary.json", summary)
    checks = {
        "status": "PASS" if passed == len(proposal_frame) else "FAIL",
        "all_events_return_below_boundary": bool(passed == len(proposal_frame)),
        "all_actions_require_human_approval": bool(proposal_frame["human_approval_required"].all()),
        "no_direct_controller_commands": bool((~proposal_frame["direct_controller_command"]).all()),
    }
    write_json(OUTPUT_DIR / "module_check.json", checks)
    if checks["status"] != "PASS":
        raise AssertionError(checks)

    build_report(proposal_frame, recovery_frame)
    print("MODULE 5 PASS - supervised adjustment and consolidated report generated")
    print(proposal_frame.to_string(index=False))


if __name__ == "__main__":
    main()
