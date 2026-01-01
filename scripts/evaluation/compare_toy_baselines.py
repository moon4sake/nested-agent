#!/usr/bin/env python

import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from exps_research.unified_framework.math_utils.qwen_math_grader import math_equal
from exps_research.unified_framework.math_utils.qwen_math_parser import extract_answer

DATASET_DIR = Path("data_processor/math_dataset/test")
DATASETS = sorted(DATASET_DIR.glob("*.json"))

BASELINE_LOG_FOLDERS = {
    "baseline_initial": Path("training_outputs/baselines/baseline_initial/qa_results"),
    "baseline_agent_distilled": Path("training_outputs/qwen-0.5B-instruct/agent_toy/qa_results"),
}

SUBNET_CONFIGS = [
    "first_half",
    "second_half",
    "stepping_half",
]


def load_dataset_answers(dataset_path: Path) -> Dict[int, str]:
    data = json.loads(dataset_path.read_text())
    return {row["id"]: row["answer"] for row in data.get("examples", [])}


def accuracy_from_scored_jsonl(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    total = 0
    correct = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        total += 1
        if row.get("score"):
            correct += 1
    accuracy = correct / total if total else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def accuracy_from_generation_json(path: Path, answers: Dict[int, str]) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    results = json.loads(path.read_text())
    total = 0
    correct = 0
    for row in results:
        example_answer = answers.get(row["id"])
        if example_answer is None:
            continue
        predicted = row.get("answer", "")
        if isinstance(predicted, str) and "boxed" in predicted:
            predicted = extract_answer(predicted)
        correct += int(math_equal(str(predicted), str(example_answer), timeout=True))
        total += 1
    accuracy = correct / total if total else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def find_scored_file(folder: Path, dataset_name: str) -> Optional[Path]:
    dataset_dir = folder / dataset_name / "evaluations"
    if not dataset_dir.exists():
        return None
    candidates = sorted(dataset_dir.glob("*_scored.jsonl"))
    if not candidates:
        return None
    return candidates[-1]


def summarize_agent_baseline(log_folder: Path) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    accuracies: List[float] = []
    for dataset_path in DATASETS:
        dataset_name = f"{dataset_path.stem}_test"
        scored_file = find_scored_file(log_folder, dataset_name)
        metrics = accuracy_from_scored_jsonl(scored_file) if scored_file else None
        if metrics is None:
            results[dataset_path.stem] = {"accuracy": 0.0, "correct": 0, "total": 0}
        else:
            results[dataset_path.stem] = metrics
            accuracies.append(metrics["accuracy"])
    avg_accuracy = mean(accuracies) if accuracies else 0.0
    results["average_accuracy"] = avg_accuracy
    return results


def summarize_subnet_baseline(prefix: str) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    overall_accuracies: List[float] = []
    for dataset_path in DATASETS:
        answers = load_dataset_answers(dataset_path)
        config_metrics = {}
        per_config_accuracies = []
        for config in SUBNET_CONFIGS:
            result_path = Path(
                f"training_outputs/nested_subnet/{prefix}_{config}_{dataset_path.stem}.json"
            )
            metrics = accuracy_from_generation_json(result_path, answers)
            if metrics is None:
                metrics = {"accuracy": 0.0, "correct": 0, "total": 0}
            config_metrics[config] = metrics
            per_config_accuracies.append(metrics["accuracy"])
        avg_accuracy = mean(per_config_accuracies) if per_config_accuracies else 0.0
        results[dataset_path.stem] = {
            "accuracy": avg_accuracy,
            "configurations": config_metrics,
        }
        overall_accuracies.append(avg_accuracy)
    results["average_accuracy"] = mean(overall_accuracies) if overall_accuracies else 0.0
    return results


def main() -> None:
    summary = {
        "baseline_initial": summarize_agent_baseline(BASELINE_LOG_FOLDERS["baseline_initial"]),
        "baseline_agent_distilled": summarize_agent_baseline(
            BASELINE_LOG_FOLDERS["baseline_agent_distilled"]
        ),
        "subnet_only": summarize_subnet_baseline("subnet_only"),
        "joint_preserve": summarize_subnet_baseline("joint_preserve"),
    }

    output_dir = Path("training_outputs/baseline_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote baseline comparison summary to {output_path}")


if __name__ == "__main__":
    main()
