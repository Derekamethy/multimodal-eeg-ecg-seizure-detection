import json
import numpy as np
import pandas as pd
import pytest

from multimodal_seizure.metrics import evaluate_record


def fixture(high=(), events=((20, 30),), uncertain=(), duration=65):
    times = np.arange(10, duration, 5, dtype=float)
    frame = pd.DataFrame(dict(anchor_s=times, score=[float(t in high) for t in times],
                              eeg_available=True, file_name="one.edf"))
    refs = pd.DataFrame([dict(event_id=f"e{i}", onset_relative_s=a, offset_relative_s_canonical=b,
                              uncertain_intervals_json=json.dumps(uncertain) if i == 0 else "[]")
                         for i, (a, b) in enumerate(events)])
    return frame, refs


# Expected counts are explicit: matched, unmatched (including duplicates), duplicates,
# state-overlapped seizures, and declaration-minus-onset delays in seconds.
@pytest.mark.parametrize("high,events,expected", [
    ([25], [(20, 30)], (1, 0, 0, 1, [5])),
    ([], [(20, 30)], (0, 0, 0, 0, [])),
    ([10], [(30, 40)], (0, 1, 0, 0, [])),
    ([10, 15, 20, 25], [(20, 30)], (0, 1, 0, 1, [])),
    ([10, 25], [(10, 40)], (1, 1, 1, 1, [0])),
    ([20, 45], [(20, 30), (40, 50)], (2, 0, 0, 2, [0, 5])),
    (list(range(10, 55, 5)), [(20, 30), (40, 50)], (0, 1, 0, 2, [])),
    ([35], [(20, 30)], (0, 1, 0, 0, [])),
    ([20], [(20, 30)], (1, 0, 0, 1, [0])),
    ([30], [(20, 30)], (0, 1, 0, 0, [])),
    ([], [(20, 30), (40, 50)], (0, 0, 0, 0, [])),
    ([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], [(10, 20), (30, 40)], (1, 0, 0, 2, [0])),
])
def test_hand_calculated_event_matching(high, events, expected):
    frame, refs = fixture(high, events)
    metrics, alarms, matches, exposure = evaluate_record(frame, refs, 65, .5)
    detected, false, duplicate, overlap, delays = expected
    assert (metrics["detected"], metrics["false_declarations"], metrics["duplicate_declarations"], metrics["overlap_detected"]) == expected[:4]
    assert [m["delay_s"] for m in matches if m["detected"]] == delays
    assert metrics["missed"] == len(events) - detected
    assert metrics["delay_n"] == detected
    assert metrics["evaluable_s"] == 55
    assert metrics["far_per_h"] == pytest.approx(false * 3600 / 55)
    if not detected:
        assert np.isnan(metrics["delay_median_s"])


def test_one_low_bridged_two_lows_close_at_actual_decision():
    frame, refs = fixture([10, 20])
    metrics, alarms, _, _ = evaluate_record(frame, refs, 65, .5)
    assert len(alarms) == 1
    assert alarms[0]["declaration_s"] == 10
    assert alarms[0]["end_s"] == 30  # low15 bridged; low25 and low30 close.
    assert alarms[0]["termination_reason"] == "two_below"
    assert metrics["active_alarm_s"] == 20
    assert metrics["warning_fraction"] == pytest.approx(20 / 55)


def test_uncertainty_between_decisions_breaks_alarm_without_time_compression():
    frame, refs = fixture([10, 15, 20], uncertain=[[12, 13]])
    metrics, alarms, _, exposure = evaluate_record(frame, refs, 65, .5)
    assert [(a["declaration_s"], a["end_s"]) for a in alarms] == [(10, 12), (15, 30)]
    assert exposure == [[10, 12], [13, 65]]
    assert metrics["uncertain_s"] == 1
    assert metrics["startup_s"] == 10
    assert metrics["evaluable_s"] == 54
    assert metrics["other_unavailable_s"] == 0


def test_missing_decision_acquisition_gap_and_unavailable_are_not_negatives():
    frame, refs = fixture([10, 15, 20])
    for changed, kwargs in (
        (frame.loc[frame.anchor_s != 15], {}),
        (frame.assign(eeg_available=frame.anchor_s != 15), {}),
        (frame, {"acquisition_gaps": [[15, 20]]}),
    ):
        metrics, alarms, _, exposure = evaluate_record(changed, refs, 65, .5, **kwargs)
        assert [a["declaration_s"] for a in alarms] == [10, 20]
        assert alarms[0]["end_s"] == 15
        assert metrics["other_unavailable_s"] == 5
        assert metrics["evaluable_s"] == 50


def test_record_boundaries_and_irregular_last_interval():
    frame, refs = fixture([60], duration=63)
    metrics, alarms, _, _ = evaluate_record(frame, refs, 63, .5)
    assert alarms[0]["end_s"] == 63
    assert metrics["evaluable_s"] == 53  # not eleven rows * five seconds
    assert metrics["active_alarm_s"] == 3
    both = pd.concat([frame, frame.assign(file_name="two.edf")])
    with pytest.raises(ValueError, match="parent EDF"):
        evaluate_record(both, refs, 63, .5)


def test_threshold_equality_and_actual_timestamp_latency():
    frame, refs = fixture(events=[(21, 32)])
    frame.loc[frame.anchor_s == 25, "score"] = .5
    metrics, alarms, matches, _ = evaluate_record(frame, refs, 65, .5)
    assert alarms[0]["declaration_s"] == 25
    assert matches[0]["delay_s"] == 4  # not support-start15 or center20
    assert metrics["delay_median_s"] == 4
