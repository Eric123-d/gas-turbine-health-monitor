"""Module 4: rank standardized sensor changes at the first forecast alarm."""

import numpy as np

from module1_data import FEATURES, load_and_split
from module2_baseline import train_baseline
from module3_kalman import test_and_plot, train_kalman


def diagnose(events, baseline):
    results = []
    for event in events:
        frame = event["frame"]; start = event["start"]; alarm = event["alarm_index"]
        reference = frame[FEATURES].iloc[max(0, start - 5):start].median().to_numpy(dtype=float)
        current = frame[FEATURES].iloc[alarm].to_numpy(dtype=float)
        scores = np.abs((current - reference) / baseline["scaler"].scale_)
        diagnosed = FEATURES[int(np.argmax(scores))]
        results.append({"duration": event["duration"], "diagnosed": diagnosed,
                        "truth": event["feature"], "correct": diagnosed == event["feature"]})
    correct = sum(item["correct"] for item in results)
    if correct != len(results):
        raise AssertionError("root-cause validation failed")
    print(f"Module 4 PASS | synthetic root cause={correct}/{len(results)}")
    return results


def main():
    _, baseline_train, matrix_train, test = load_and_split()
    baseline = train_baseline(baseline_train, test)
    kalman = train_kalman(matrix_train, baseline)
    events = test_and_plot(test, baseline, kalman)
    diagnose(events, baseline)
    print("ALL FOUR MODULES PASS | only residual_alarm.png is written")


if __name__ == "__main__":
    main()
