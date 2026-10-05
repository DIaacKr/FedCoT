import copy
import csv
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import click
import numpy as np
import torch
from peft import AutoPeftModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from answer_extractor import compute_reward, extract_answer, match_choice_answer
from data_preparation import get_latest_version_dir, parse_dataset_versions


logger = logging.getLogger(__name__)


class RewardModelSelector:
    def __init__(
        self,
        model_path: str,
        batch_size: int = 64,
        max_length: int = 512,
        model_name: str = None,
        random_seed: int = 42,
    ):
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_name = model_name or self.model_path.name
        self.random_seed = random_seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if ((self.model_path / "adapter_config.json").exists()
                or (self.model_path / "adapter_model.safetensors").exists()):
            self.model = AutoPeftModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=1,
                ignore_mismatched_sizes=True,
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=1,
                ignore_mismatched_sizes=True,
            )

        self.model.to(self.device)
        if self.device.type == "cuda":
            self.model.half()
        self.model.eval()

    def score_responses(self, question: str, responses: List[str]) -> List[float]:
        texts = [f"Q: {question}\nA: {response}" for response in responses]
        scores = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                inputs = self.tokenizer(
                    texts[start:start + self.batch_size],
                    padding=True,
                    truncation=True,
                    # Preserve the experiment's actual scoring window.
                    max_length=4096,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                logits = self.model(**inputs).logits
                scores.extend(torch.sigmoid(logits).view(-1).float().cpu().tolist())
        return scores

    def select_best_candidate(self, question: str, candidates: List[Dict]) -> Tuple[int, List[float]]:
        scores = self.score_responses(question, [item["response_text"] for item in candidates])
        return int(np.argmax(scores)), scores

    def get_method_info(self) -> Dict:
        return {
            "method": "fedcot_reward_model",
            "model_path": str(self.model_path),
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "tokenization_max_length": 4096,
        }


def recalculate_dataset_correctness(test_data: List[Dict]) -> List[Dict]:
    for sample in test_data:
        options = sample.get("options", {})
        for candidate in sample.get("candidates", []):
            extracted = extract_answer(candidate["response_text"])
            answer = match_choice_answer(extracted, options)
            reward = compute_reward(answer, sample)
            candidate["extracted_answer"] = answer
            candidate["reward"] = reward
            candidate["is_correct"] = reward == 1.0
    return test_data


def load_dataset_test_data(
    model_output_dir: str,
    dataset_name: str,
    version: str = "latest",
) -> Tuple[List[Dict], Dict, str]:
    base_dir = Path(model_output_dir) / f"{dataset_name}_best_of_n" / "test"
    data_dir = get_latest_version_dir(base_dir) if version == "latest" else base_dir / (
        version if version.startswith("version_") else f"version_{version}"
    )
    actual_version = data_dir.name if data_dir.name.startswith("version_") else "base"
    test_path = data_dir / "test_best_of_n.json"

    with test_path.open("r", encoding="utf-8") as source:
        test_data = recalculate_dataset_correctness(json.load(source))

    total = len(test_data)
    first_correct = sum(
        bool(sample.get("candidates")) and sample["candidates"][0].get("is_correct", False)
        for sample in test_data
    )
    any_correct = sum(
        any(candidate.get("is_correct", False) for candidate in sample.get("candidates", []))
        for sample in test_data
    )
    pool_stats = {
        "candidate_pass_at_1": first_correct / total if total else 0.0,
        "candidate_pass_at_k": any_correct / total if total else 0.0,
    }
    metadata = {
        "dataset_name": dataset_name,
        "version_requested": version,
        "version_actual": actual_version,
        "data_path": str(test_path),
        "sample_count": total,
        **pool_stats,
    }
    return test_data, metadata, actual_version


def evaluate_selector_performance(
    test_data: List[Dict],
    selector: RewardModelSelector,
) -> Tuple[Dict, List[Dict]]:
    correct_count = 0
    start_time = time.time()

    for sample in test_data:
        candidates = sample.get("candidates", [])
        if not candidates:
            continue

        best_index, scores = selector.select_best_candidate(sample["question"], candidates)
        for candidate, score in zip(candidates, scores):
            candidate["reward_model_score"] = score
        sample["selected_candidate_id"] = best_index
        correct_count += bool(candidates[best_index].get("is_correct", False))

    elapsed = time.time() - start_time
    total = len(test_data)
    return {
        "correct": correct_count,
        "total": total,
        "accuracy": correct_count / total if total else 0.0,
        "evaluation_time": elapsed,
    }, test_data


def run_evaluation(
    model_output_dir: str,
    selector: RewardModelSelector,
    dataset_names: List[str],
    dataset_versions: Dict[str, str],
) -> Tuple[Dict, Dict]:
    results = {}
    actual_versions = {}
    for dataset_name in dataset_names:
        version = dataset_versions.get(dataset_name, "latest")
        test_data, metadata, actual_version = load_dataset_test_data(
            model_output_dir, dataset_name, version
        )
        metrics, detailed_results = evaluate_selector_performance(test_data, selector)
        actual_versions[dataset_name] = actual_version
        results[dataset_name] = {
            "test_samples": metrics["total"],
            "version": actual_version,
            "candidate_pass_at_1": metadata["candidate_pass_at_1"],
            "candidate_pass_at_k": metadata["candidate_pass_at_k"],
            "selector_accuracy": metrics["accuracy"],
            "evaluation_time": metrics["evaluation_time"],
            "detailed_results": detailed_results,
        }
    return results, actual_versions


def save_results(results: Dict, output_path: Path, metadata: Dict) -> None:
    output_path = Path(output_path)
    summary = copy.deepcopy(results)
    for dataset_name, result in results.items():
        if "detailed_results" in result:
            detailed_path = output_path.parent / f"{dataset_name}_with_rewards.json"
            with detailed_path.open("w", encoding="utf-8") as destination:
                json.dump(result["detailed_results"], destination, ensure_ascii=False, indent=2)
            del summary[dataset_name]["detailed_results"]

    with output_path.open("w", encoding="utf-8") as destination:
        json.dump({"results": summary, "metadata": metadata}, destination, ensure_ascii=False, indent=2)

    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(("dataset", "selector_accuracy", "candidate_pass_at_1", "candidate_pass_at_k"))
        for dataset_name, result in summary.items():
            if "test_samples" in result:
                writer.writerow((
                    dataset_name,
                    f"{result['selector_accuracy']:.4f}",
                    f"{result['candidate_pass_at_1']:.4f}",
                    f"{result['candidate_pass_at_k']:.4f}",
                ))


def calculate_summary_statistics(results: Dict) -> Dict:
    dataset_results = [result for result in results.values() if "test_samples" in result]
    if dataset_results:
        results["macro_average"] = {
            key: sum(result[key] for result in dataset_results) / len(dataset_results)
            for key in ("selector_accuracy", "candidate_pass_at_1", "candidate_pass_at_k")
        }
    return results


@click.command()
@click.option("--dataset_names", required=True, help="Comma-separated dataset names.")
@click.option("--model_output_dir", required=True, help="Directory containing candidate files.")
@click.option("--dataset_versions", default="", help="One version or one version per dataset.")
@click.option("--reward_model_path", required=True, help="Trained FedCoT reward model path.")
@click.option("--batch_size", default=512, type=int)
@click.option("--max_length", default=512, type=int)
@click.option("--random_seed", default=42, type=int)
def evaluate_candidate_selector(
    dataset_names: str,
    model_output_dir: str,
    dataset_versions: str,
    reward_model_path: str,
    batch_size: int,
    max_length: int,
    random_seed: int,
) -> None:
    datasets = [name.strip() for name in dataset_names.split(",") if name.strip()]
    versions = parse_dataset_versions(dataset_names, dataset_versions)
    selector = RewardModelSelector(reward_model_path, batch_size, max_length, random_seed=random_seed)
    results, actual_versions = run_evaluation(model_output_dir, selector, datasets, versions)
    results = calculate_summary_statistics(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/evaluations") / Path(model_output_dir).name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fedcot_evaluation.json"
    save_results(results, output_path, {
        "model_output_dir": model_output_dir,
        "dataset_names": datasets,
        "dataset_versions_requested": versions,
        "dataset_versions_used": actual_versions,
        "selector_info": selector.get_method_info(),
    })
    print(output_path)


if __name__ == "__main__":
    evaluate_candidate_selector()
