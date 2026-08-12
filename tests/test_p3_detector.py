import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from p3_byzantine_detection import (
    Thresholds,
    apply_detector,
    build_sustain_flags,
    identify_live_rounds_by_client,
    identify_live_rounds,
)


def test_identify_live_rounds_excludes_all_zero_rounds():
    df = pd.DataFrame(
        {
            "round": [1, 1, 2, 2, 3, 3],
            "client_id": [1, 2, 1, 2, 1, 2],
            "shapley_value": [0.1, -0.2, 0.0, 0.0, 0.3, 0.4],
        }
    )

    live_rounds = identify_live_rounds(df)

    assert live_rounds[1] is True
    assert live_rounds[2] is False
    assert live_rounds[3] is True


def test_build_sustain_flags_uses_only_live_rounds():
    anomalies = [True, False, True, True]
    live_mask = [True, False, True, True]

    flags = build_sustain_flags(anomalies, live_mask, sustain_ratio=0.5)

    assert flags == [False, False, True, True]


def test_apply_detector_uses_baseline_live_rounds():
    baseline_df = pd.DataFrame(
        {
            "round": [1, 2, 3, 4, 1, 2, 3, 4],
            "client_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "shapley_value": [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.0, 0.0],
        }
    )
    detection_df = pd.DataFrame(
        {
            "round": [1, 2, 3, 4, 1, 2, 3, 4],
            "client_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "shapley_value": [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0],
        }
    )
    thresholds = Thresholds(round_means={1: 0.0, 2: 0.0, 3: 0.0}, round_stds={1: 1.0, 2: 1.0, 3: 1.0})

    result_df = apply_detector(
        detection_df,
        thresholds,
        window_size=3,
        sustain_ratio=0.5,
        live_rounds_by_client=identify_live_rounds_by_client(baseline_df),
    )

    client1_round4 = result_df[(result_df["client_id"] == 1) & (result_df["round"] == 4)]["flagged_status"].iat[0]
    client2_round2 = result_df[(result_df["client_id"] == 2) & (result_df["round"] == 2)]["flagged_status"].iat[0]

    assert client1_round4 == 1
    assert client2_round2 == 1
