"""Evaluation of (1) the rule-based classifier on the labelled synthetic dataset and
(2) optionally, speed of the pretrained person detector on a user-supplied video.

Run from the project root:
    python -m evaluation.evaluate                      # classifier only
    python -m evaluation.evaluate --video path/to.mp4  # + detector speed benchmark

The two result sets measure different things and must never be mixed:
  * classifier metrics - does the rule engine apply the access policy correctly (labels from the policy)?
  * detector benchmark - how fast is YOLO on this machine? (There is no ground-truth person annotation
    in this project, so detector ACCURACY is not measured.)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from config.settings import DEFAULT_CONFIDENCE, REPORTS_DIR
from detection.intrusion_engine import IntrusionEngine
from detection.zone_manager import BuildingConfig, ZoneManager
from evaluation.metrics import classification_metrics, latency_summary
from simulation.synthetic_dataset import LabelledScenario, build_dataset, save_csv

log = logging.getLogger(__name__)


def evaluate_classifier(scenarios: list[LabelledScenario]) -> tuple[pd.DataFrame, dict]:
    """Run each scenario on a fresh engine (default building config) and compare with expected outcomes."""
    rows = []
    for sc in scenarios:
        engine = IntrusionEngine(ZoneManager(BuildingConfig.default()))
        alerts = [a for obs in sc.observations for a in engine.evaluate(obs).alerts]
        rules = tuple(a.rule_id for a in alerts)
        rows.append({
            "scenario_id": sc.scenario_id, "scenario": sc.name, "origin": sc.origin,
            "expected_suspicious": sc.expected_suspicious, "predicted_suspicious": bool(alerts),
            "expected_alerts": sc.expected_alerts, "actual_alerts": len(alerts),
            "alert_count_match": sc.expected_alerts == len(alerts),
            "expected_rules": ",".join(sc.expected_rules), "actual_rules": ",".join(rules),
        })
    df = pd.DataFrame(rows)
    m = classification_metrics(df.expected_suspicious.tolist(), df.predicted_suspicious.tolist())
    m["alert_count_match_rate"] = float(df.alert_count_match.mean()) if len(df) else 0.0
    hw = df[df.origin == "hand_written"]
    m["hand_written_n"] = len(hw)
    m["hand_written_rule_sequence_match_rate"] = float((hw.expected_rules == hw.actual_rules).mean()) if len(hw) else 0.0
    return df, m


def benchmark_detector(video: Path, confidence: float, stride: int, max_frames: int, device: str) -> dict:
    from detection.person_detector import PersonDetector, iter_frames, open_video, video_info
    det = PersonDetector(device=device)
    det.load()
    cap = open_video(video)
    info = video_info(cap)
    times, counts = [], []
    t_wall = time.perf_counter()
    try:
        for _, _, frame in iter_frames(cap, stride=stride, max_frames=max_frames):
            dets, ms = det.detect(frame, confidence)
            times.append(ms)
            counts.append(len(dets))
    finally:
        cap.release()
    wall = time.perf_counter() - t_wall
    # the very first call includes one-off warm-up; report it separately from steady state
    summary = latency_summary(times[1:] if len(times) > 1 else times, wall, len(times))
    return {"video_file": Path(video).name, "resolution": f"{info['width']}x{info['height']}", "device": device,
            "confidence": confidence, "stride": stride, "frames_processed": len(times),
            "first_frame_ms_warmup": times[0] if times else None,
            "mean_people_per_frame": (sum(counts) / len(counts)) if counts else 0.0, **summary,
            "note": "Speed only. No ground-truth annotations, so detection accuracy is not measured."}


def save_charts(df: pd.DataFrame, m: dict, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    fig, ax = plt.subplots(figsize=(4, 3.6))
    grid = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    ax.imshow(grid, cmap="Blues")
    ax.set_xticks([0, 1], ["Normal", "Suspicious"]); ax.set_yticks([0, 1], ["Normal", "Suspicious"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Expected"); ax.set_title("Rule classifier: confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, grid[i][j], ha="center", va="center", fontsize=14,
                    color="white" if grid[i][j] > max(map(max, grid)) / 2 else "black")
    fig.tight_layout(); p = out_dir / "confusion_matrix.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    keys = ["accuracy", "precision", "recall", "f1", "false_positive_rate"]
    ax.bar(["Acc", "Prec", "Recall", "F1", "FPR"], [m[k] for k in keys], color="#38bdf8")
    ax.set_ylim(0, 1.05); ax.set_title("Rule classifier metrics (synthetic dataset)")
    fig.tight_layout(); p = out_dir / "classifier_metrics.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)
    return paths


def write_markdown(m: dict, bench: dict | None, path: Path) -> None:
    lines = ["# EdgeGuard AI - evaluation results (generated)", "",
             "## A. Rule-based intrusion classifier (synthetic labelled scenarios)", "",
             "Labels come from the written access policy; these numbers verify the rule engine, "
             "not real-world detection accuracy.", "",
             "| Metric | Value |", "|---|---|", f"| Scenarios | {m['n']} |",
             f"| TP / FP / TN / FN | {m['tp']} / {m['fp']} / {m['tn']} / {m['fn']} |",
             f"| Accuracy | {m['accuracy']:.3f} |", f"| Precision | {m['precision']:.3f} |",
             f"| Recall | {m['recall']:.3f} |", f"| F1-score | {m['f1']:.3f} |",
             f"| False-positive rate | {m['false_positive_rate']:.3f} |",
             f"| Alert-count match rate | {m['alert_count_match_rate']:.3f} |", "",
             "## B. Pretrained person detector (speed only)", ""]
    if bench:
        lines += ["| Item | Value |", "|---|---|"] + [f"| {k} | {v:.2f} |" if isinstance(v, float) else f"| {k} | {v} |"
                                                        for k, v in bench.items()]
    else:
        lines.append("_Not measured yet. Run `python -m evaluation.evaluate --video <file>` or use the Video Monitor tab._")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(seed: int = 42, per_scenario: int = 10, video: Path | None = None, confidence: float = DEFAULT_CONFIDENCE,
        stride: int = 3, max_frames: int = 300, device: str = "cpu", out_dir: Path = REPORTS_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = build_dataset(seed, per_scenario)
    save_csv(scenarios, out_dir / "synthetic_dataset.csv")
    df, m = evaluate_classifier(scenarios)
    df.to_csv(out_dir / "classifier_results.csv", index=False)
    bench = benchmark_detector(video, confidence, stride, max_frames, device) if video else None
    if bench:
        (out_dir / "detector_benchmark.json").write_text(json.dumps(bench, indent=2), encoding="utf-8")
    (out_dir / "evaluation_summary.json").write_text(json.dumps({"seed": seed, "classifier": m}, indent=2), encoding="utf-8")
    save_charts(df, m, out_dir)
    write_markdown(m, bench, out_dir / "evaluation_summary.md")
    return {"classifier": m, "detector": bench, "table": df}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-scenario", type=int, default=10)
    ap.add_argument("--video", type=Path, help="optional video for the detector speed benchmark")
    ap.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    out = run(a.seed, a.per_scenario, a.video, a.confidence, a.stride, a.max_frames, a.device)
    print(json.dumps({"classifier": out["classifier"], "detector": out["detector"]}, indent=2, default=str))
    mism = out["table"][~out["table"].alert_count_match | (out["table"].expected_suspicious != out["table"].predicted_suspicious)]
    print(f"\nMismatching scenarios: {len(mism)}")
    if len(mism):
        print(mism.to_string())


if __name__ == "__main__":
    main()
