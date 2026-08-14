# Simplified Gas Turbine Health Monitor

This is a compact interview project that demonstrates early fault warning for a gas turbine. The repository contains only ten files and does not save training traces, intermediate matrices, or duplicate reports.

The single `data.csv` file is based on the public GE Frame 9E operating dataset. Because the public data has no authoritative fault labels, Module 1 removes non-numeric values, physically impossible values, duplicate rows, isolated spikes, and large-residual outliers. The remaining rows are treated as a strictly screened healthy reference set. The original sampling interval is about 240 seconds; the code does not create artificial 1 Hz samples.

## Five modules

1. `module1_data.py` cleans the data and makes a chronological 40%/30%/30% split.
2. `module2_baseline.py` trains an Elastic Net model on the first 40%. It calculates the non-negative absolute residual and a one-sided 10-SD safety boundary.
3. `module3_kalman.py` injects several gradual fault patterns into the middle 30% and learns the A, B, C, bias, Q, and R terms. At each new sample, the adaptive Kalman filter updates only the state and P, then forecasts no more than 600 sampling steps.
4. `module4_root_cause.py` compares standardized sensor changes at the first alarm and identifies the likely source.
5. `module5_adjustment.py` verifies a supervised return toward the pre-fault reference state. It never sends a real control command.

## Run

```powershell
pip install -r requirements.txt
./run_all.ps1
```

Each module prints a short validation result. The current Elastic Net scores are approximately 0.985 training R² and 0.855 final-test R². Fault tests cover 10, 30, 100, 300, and 600 sampling steps. For this dataset, 600 steps equal about 144,000 seconds. An alarm requires two consecutive residual increases and a Kalman forecast that crosses the boundary within 600 steps; the red alarm point normally appears 2–3 samples after fault onset. The only saved result is `residual_alarm.png`.

The synthetic faults, statistical boundary, and adjustment logic are simplified demonstrations, not OEM protection limits or field-performance claims.
