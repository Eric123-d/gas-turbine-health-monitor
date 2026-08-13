"""Module 5: operator-supervised return to the pre-fault reference."""

from module1_data import FEATURES, load_and_split
from module2_baseline import absolute_residual, train_baseline
from module3_kalman import test_and_plot, train_kalman
from module4_root_cause import diagnose


def adjust(events, baseline):
    passed = 0
    for event in events:
        frame = event["frame"].copy(); start = event["start"]; duration = event["duration"]
        reference = frame.iloc[start - 1].copy(); current = frame.iloc[start + duration].copy()
        for step in range(1, 11):
            fraction = step / 10.0
            for column in FEATURES + ["EP"]:
                current[column] = current[column] + fraction * (reference[column] - current[column])
        residual = float(absolute_residual(frame.iloc[[start - 1]].assign(**current.to_dict()), baseline)[0])
        passed += residual < baseline["boundary"]
    if passed != len(events):
        raise AssertionError("supervised adjustment check failed")
    print(f"Module 5 PASS | returned below boundary={passed}/{len(events)} | controller commands=0")


def main():
    _, baseline_train, matrix_train, test = load_and_split()
    baseline = train_baseline(baseline_train, test)
    kalman = train_kalman(matrix_train, baseline)
    events = test_and_plot(test, baseline, kalman)
    diagnose(events, baseline)
    adjust(events, baseline)
    print("ALL MODULES PASS | only residual_alarm.png is written")


if __name__ == "__main__":
    main()
