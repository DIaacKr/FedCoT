import re
from typing import Dict, Optional


def extract_answer(response_text: str) -> str:
    answer = extract_hash_content(response_text)
    return answer.strip() if answer else ""


def extract_hash_content(text: str) -> Optional[str]:
    index = text.rfind("####")
    if index < 0:
        return None

    content = text[index + 4:].strip()
    if "\n" in content:
        content = content.split("\n", 1)[0].strip()
    return content or None


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    trailing_punctuation = ".,;:!?\'\""
    while text and text[-1] in trailing_punctuation:
        text = text[:-1].strip()
    return " ".join(text.split())


def match_choice_answer(extracted_text: str, options_dict: Dict[str, str]) -> str:
    if not extracted_text or not options_dict:
        return ""
    match = re.match(r"^([A-Z])\.\s*(.+)$", extracted_text.strip())
    if not match:
        return ""

    letter, extracted_content = match.groups()
    expected_content = options_dict.get(letter)
    if expected_content is not None and _normalize_text(extracted_content) == _normalize_text(expected_content):
        return letter
    return ""


def compute_reward(extracted_answer: str, ground_truth_item: Dict) -> float:
    if extracted_answer and extracted_answer.upper() == ground_truth_item["answer_letter"].upper():
        return 1.0
    return 0.0
