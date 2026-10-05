import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_dataset_versions(dataset_names: str, dataset_versions: Optional[str] = None) -> Dict[str, str]:
    datasets = [name.strip() for name in dataset_names.split(",") if name.strip()]
    if not dataset_versions:
        return {name: "latest" for name in datasets}

    versions = [version.strip() for version in dataset_versions.split(",")]
    if len(versions) == 1:
        return {name: versions[0] for name in datasets}
    if len(versions) == len(datasets):
        return dict(zip(datasets, versions))
    raise ValueError("Provide one dataset version or one version per dataset.")


def get_latest_version_dir(base_dir: Path) -> Path:
    versions = []
    if base_dir.exists():
        for path in base_dir.iterdir():
            if path.is_dir() and path.name.startswith("version_"):
                try:
                    versions.append((int(path.name.split("_")[1]), path))
                except (IndexError, ValueError):
                    continue
    return max(versions, default=(0, base_dir), key=lambda item: item[0])[1]


def get_training_data_dir(base_dir: Path, version: str) -> Path:
    if version == "latest":
        return get_latest_version_dir(base_dir)
    return base_dir / (version if version.startswith("version_") else f"version_{version}")


def extract_nli_pair(question: str, response: str) -> Optional[Tuple[str, str]]:
    parts = response.split("####")
    if len(parts) < 2:
        return None

    premise = parts[0].strip()
    answer = " ".join(line.strip() for line in parts[-1].strip().split("\n") if line.strip())
    hypothesis = f"{question.strip()} The answer is {answer}."

    premise_words = premise.split()
    hypothesis_words = hypothesis.split()
    if len(premise_words) > 200:
        premise = " ".join(premise_words[-200:])
    if len(hypothesis_words) > 150:
        hypothesis = " ".join(hypothesis_words[-150:])
    if not premise or not hypothesis:
        return None
    return premise, hypothesis


class EntailmentScorer:
    def __init__(self, model_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        id2label = self.model.config.id2label
        if len(id2label) < 3:
            raise ValueError("The soft-reward scorer requires an NLI model with at least three labels.")
        mapping = {}
        for index, label in id2label.items():
            label = label.lower()
            if "entail" in label:
                mapping[index] = "entailment"
            elif "contradict" in label:
                mapping[index] = "contradiction"
            elif "neutral" in label:
                mapping[index] = "neutral"
        self.entailment_index = (
            next(index for index, label in mapping.items() if label == "entailment")
            if len(mapping) == 3 else 1
        )

    def score(self, premise: str, hypothesis: str) -> float:
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            return torch.softmax(logits, dim=-1)[0, self.entailment_index].item()


def filter_training_candidates(data: List[Dict], rng: random.Random) -> List[Dict]:
    filtered = []
    for item in data:
        accuracy = float(item.get("accuracy", -1))
        if accuracy >= 1.0 or accuracy <= 0.0:
            continue

        candidates = item.get("candidates", [])
        if not candidates:
            continue

        correct = [candidate for candidate in candidates if candidate.get("is_correct", False)]
        wrong_with_answer = [
            candidate
            for candidate in candidates
            if not candidate.get("is_correct", False)
            and candidate.get("extracted_answer") is not None
            and str(candidate.get("extracted_answer")).strip()
        ]
        wrong_without_answer = [
            candidate
            for candidate in candidates
            if not candidate.get("is_correct", False)
            and (candidate.get("extracted_answer") is None
                 or not str(candidate.get("extracted_answer")).strip())
        ]
        wrong = wrong_with_answer + wrong_without_answer
        if not correct or not wrong:
            continue

        if len(correct) == 1:
            selected = [correct[0], rng.choice(wrong)]
        elif wrong_with_answer and wrong_without_answer:
            selected = rng.sample(correct, 2)
            selected.extend((rng.choice(wrong_with_answer), rng.choice(wrong_without_answer)))
        else:
            selected = [rng.choice(correct), rng.choice(wrong)]

        filtered_item = item.copy()
        filtered_item["candidates"] = selected
        filtered_item["num_candidates"] = len(selected)
        filtered.append(filtered_item)
    return filtered


def assign_soft_rewards(data: List[Dict], scorer: EntailmentScorer) -> List[Dict]:
    for item in data:
        question = item["question"]
        for candidate in item["candidates"]:
            is_correct = bool(candidate.get("is_correct", False))
            pair = extract_nli_pair(question, candidate.get("response_text", ""))
            entailment = scorer.score(*pair) if pair else 0.0
            candidate["original_reward"] = candidate.get("reward", 0.0)
            candidate["reward"] = round((0.5 if is_correct else 0.0) + 0.5 * entailment, 4)
            candidate["nli_entailment"] = round(entailment, 4)
            candidate["has_valid_format"] = pair is not None
    return data


def prepare_training_data(
    model_output_dir: Path,
    dataset_names: List[str],
    dataset_versions: Dict[str, str],
    nli_model: str,
    seed: int,
    overwrite: bool = False,
) -> List[Path]:
    jobs = []
    for dataset_name in dataset_names:
        base_dir = model_output_dir / f"{dataset_name}_best_of_n" / "train"
        data_dir = get_training_data_dir(base_dir, dataset_versions.get(dataset_name, "latest"))
        input_path = data_dir / "train_best_of_n.json"
        output_path = data_dir / "train_best_of_n_filtered_soft.json"
        if output_path.exists() and not overwrite:
            continue
        jobs.append((input_path, output_path))

    if not jobs:
        return []

    scorer = EntailmentScorer(nli_model)
    outputs = []
    rng = random.Random(seed)
    for input_path, output_path in jobs:
        with input_path.open("r", encoding="utf-8") as source:
            data = json.load(source)
        filtered = filter_training_candidates(data, rng)
        prepared = assign_soft_rewards(filtered, scorer)
        with output_path.open("w", encoding="utf-8") as destination:
            json.dump(prepared, destination, ensure_ascii=False, indent=2)
        outputs.append(output_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare filtered FedCoT training data with NLI soft rewards.")
    parser.add_argument("--model_output_dir", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--dataset_versions", default="")
    parser.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    versions = parse_dataset_versions(args.datasets, args.dataset_versions)
    outputs = prepare_training_data(
        Path(args.model_output_dir),
        dataset_names,
        versions,
        args.nli_model,
        args.random_seed,
        args.overwrite,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
