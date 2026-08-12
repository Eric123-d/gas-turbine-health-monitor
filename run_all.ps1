$ErrorActionPreference = "Stop"

python module1_data.py
python module2_baseline.py
python module3_kalman.py
python module4_root_cause.py
python module5_adjustment.py

Write-Host "All five modules passed. Open outputs/PROJECT_REPORT.html"
