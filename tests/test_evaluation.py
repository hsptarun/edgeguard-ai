import pytest

from evaluation.evaluate import evaluate_classifier, run
from evaluation.metrics import classification_metrics, confusion_matrix, latency_summary
from simulation.synthetic_dataset import build_dataset, hand_written


def test_metrics_known_values():
    y_true = [True, True, True, False, False, False, False, True]
    y_pred = [True, True, False, True, False, False, False, True]
    m = classification_metrics(y_true, y_pred)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (3, 1, 3, 1)
    assert m["precision"] == 0.75 and m["recall"] == 0.75 and m["f1"] == 0.75
    assert m["false_positive_rate"] == 0.25 and m["accuracy"] == 0.75


def test_metrics_edge_cases():
    assert classification_metrics([], [])["f1"] == 0.0
    assert classification_metrics([False], [False])["precision"] == 0.0
    with pytest.raises(ValueError):
        confusion_matrix([True], [True, False])


def test_latency_summary():
    s = latency_summary([10, 20, 30], wall_seconds=1.0, frames=3)
    assert s["inference_ms_mean"] == 20 and round(s["inference_fps"]) == 50 and s["pipeline_fps_wall_clock"] == 3
    assert latency_summary([]) == {"frames_timed": 0}


def test_hand_written_scenarios_all_classified_correctly():
    df, m = evaluate_classifier(hand_written())
    bad = df[(df.expected_suspicious != df.predicted_suspicious) | ~df.alert_count_match]
    assert bad.empty, bad.to_string()
    assert m["hand_written_rule_sequence_match_rate"] == 1.0


def test_full_pipeline_writes_reports(tmp_path):
    out = run(seed=5, per_scenario=2, out_dir=tmp_path)
    for f in ("classifier_results.csv", "evaluation_summary.json", "evaluation_summary.md", "confusion_matrix.png",
              "synthetic_dataset.csv"):
        assert (tmp_path / f).exists()
    m = out["classifier"]
    assert m["n"] == len(build_dataset(5, 2)) and m["tp"] + m["fn"] == sum(s.expected_suspicious for s in build_dataset(5, 2))
    assert out["detector"] is None
