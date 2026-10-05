import json
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset

def get_datasets(seed, dataset_name):
    """
    Load and process datasets, converting them to multiple choice question format
    """
    rng = np.random.default_rng(seed=seed)
    
    if dataset_name == 'pubmedqa':
        dataset = load_dataset('pubmed_qa', 'pqa_labeled', split='train')
        dataset = dataset.shuffle(seed=seed)
        
        train_dataset = []
        for i in range(450):
            item = _process_pubmedqa_item(dataset[i], rng)
            train_dataset.append(item)
            
        val_dataset = []
        for i in range(450, 500):
            item = _process_pubmedqa_item(dataset[i], rng)
            val_dataset.append(item)
            
        test_dataset = []
        for i in range(500, 1000):
            item = _process_pubmedqa_item(dataset[i], rng)
            test_dataset.append(item)
            
    elif dataset_name == 'medmcqa':
        temp_dataset = load_dataset('medmcqa', split='train')
        temp_dataset = temp_dataset.shuffle(seed=seed)
        
        train_dataset = []
        num_train = 0
        for i in range(len(temp_dataset)):
            if temp_dataset[i]['exp'] is not None:
                item = _process_medmcqa_item(temp_dataset[i], rng)
                train_dataset.append(item)
                num_train += 1
                if num_train >= 2000:
                    break
                    
        temp_dataset = load_dataset('medmcqa', split='validation')
        temp_dataset = temp_dataset.shuffle(seed=seed)
        
        val_dataset = []
        for i in range(len(temp_dataset)):
            item = _process_medmcqa_item(temp_dataset[i], rng)
            val_dataset.append(item)
            
        test_dataset = []
        for i in range(len(temp_dataset)):
            item = _process_medmcqa_item(temp_dataset[i], rng)
            test_dataset.append(item)
            
    elif dataset_name == 'mmlu':
        subset_names = ['clinical_knowledge', 'college_biology', 'college_medicine', 
                       'high_school_biology', 'medical_genetics', 'professional_medicine', 'virology']
        total_dataset = []
        
        for subset in subset_names:
            dataset = load_dataset('lukaemon/mmlu', subset, trust_remote_code=True)
            for key in dataset.keys():
                for i in range(len(dataset[key])):
                    total_dataset.append(dataset[key][i])
                    
        random.shuffle(total_dataset)
        
        train_dataset = []
        for i in range(int(0.8 * len(total_dataset))):
            item = _process_mmlu_item(total_dataset[i], rng)
            train_dataset.append(item)
            
        val_dataset = []
        for i in range(int(0.8 * len(total_dataset)), int(0.9 * len(total_dataset))):
            item = _process_mmlu_item(total_dataset[i], rng)
            val_dataset.append(item)
            
        test_dataset = []
        for i in range(int(0.9 * len(total_dataset)), len(total_dataset)):
            item = _process_mmlu_item(total_dataset[i], rng)
            test_dataset.append(item)
            
    elif dataset_name == 'medqa':
        # Train split
        dataset = load_dataset('bigbio/med_qa', 'med_qa_en_4options_bigbio_qa', split='train', trust_remote_code=True)
        dataset = dataset.shuffle(seed=seed)
        
        train_dataset = []
        for i in range(len(dataset)):
            item = _process_medqa_item(dataset[i], rng)
            train_dataset.append(item)
            
        # Validation split
        dataset = load_dataset('bigbio/med_qa', 'med_qa_en_4options_bigbio_qa', split='validation', trust_remote_code=True)
        dataset = dataset.shuffle(seed=seed)
        
        val_dataset = []
        for i in range(len(dataset)):
            item = _process_medqa_item(dataset[i], rng)
            val_dataset.append(item)
            
        # Test split
        dataset = load_dataset('bigbio/med_qa', 'med_qa_en_4options_bigbio_qa', split='test', trust_remote_code=True)
        dataset = dataset.shuffle(seed=seed)
        
        test_dataset = []
        for i in range(len(dataset)):
            item = _process_medqa_item(dataset[i], rng)
            test_dataset.append(item)
            
    elif dataset_name == 'bioasq':
        dataset = []
        bioasq_root = Path(__file__).resolve().parents[1] / "data" / "bioasq"
        for year in [11, 10, 9, 8, 7]:
            year_dir = bioasq_root / f"Task{year}BGoldenEnriched"
            question_files = sorted(
                year_dir.glob(f"{year}B*_golden.json"),
                key=lambda path: int(path.stem.split("B")[-1].split("_")[0]),
            )
            if not question_files:
                raise FileNotFoundError(
                    f"No BioASQ golden files found under {year_dir}. "
                    "Download Task 7B through 11B GoldenEnriched files and place them there."
                )
            for question_file in question_files:
                with question_file.open("r", encoding="utf-8") as source:
                    questions = json.load(source)["questions"]
                dataset.extend(item for item in questions if item.get("type") == "yesno")
        random.shuffle(dataset)
        
        train_dataset = []
        for i in range(int(0.8 * len(dataset))):
            item = _process_bioasq_item(dataset[i], rng)
            train_dataset.append(item)
            
        test_dataset = []
        for i in range(int(0.8 * len(dataset)), len(dataset)):
            item = _process_bioasq_item(dataset[i], rng)
            test_dataset.append(item)
            
        val_dataset = []

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_dataset, val_dataset, test_dataset

def _process_pubmedqa_item(item, rng):
    """Process PubMedQA data items, converting to three-choice question format"""
    contexts = item['context']
    contexts_final = 'Contexts:\n'
    for j in range(len(contexts['contexts'])):
        contexts_final += contexts['labels'][j] + ': ' + contexts['contexts'][j] + '\n'
    
    base_question = contexts_final + item['question']
    final_decision = item['final_decision']
    long_answer = item['long_answer']
    
    # Define three options
    raw_options = ['yes', 'no', 'maybe']
    correct_answer = final_decision
    
    # Randomly shuffle options
    options_list = raw_options.copy()
    rng.shuffle(options_list)
    
    # Find the new position of the correct answer
    answer_idx = options_list.index(correct_answer)
    answer_letter = chr(ord('A') + answer_idx)
    answer_string = correct_answer
    
    # Create options dictionary and string
    options = {chr(ord('A') + i): opt for i, opt in enumerate(options_list)}
    options_str = "\n".join([f"{op}. {ans}" for op, ans in options.items()])
    
    # Complete question including options
    question = base_question + "\n" + options_str
    
    # Original answer format
    answer = long_answer + '\n#### ' + final_decision
    
    return {
        'question': question,
        'answer': answer,
        'answer_string': answer_string,
        'answer_letter': answer_letter,
        'answer_idx': answer_idx,
        'options': options,
        'options_str': options_str
    }

def _process_medmcqa_item(item, rng):
    """Process MedMCQA data items"""
    base_question = item['question']
    raw_options = [item['opa'], item['opb'], item['opc'], item['opd']]
    correct_idx = item['cop']
    explanation = item['exp']
    
    # Get correct answer string
    answer_string = raw_options[correct_idx]
    
    # Randomly shuffle options
    options_list = raw_options.copy()
    rng.shuffle(options_list)
    
    # Find the new position of the correct answer
    answer_idx = options_list.index(answer_string)
    answer_letter = chr(ord('A') + answer_idx)
    
    # Create options dictionary and string
    options = {chr(ord('A') + i): opt for i, opt in enumerate(options_list)}
    options_str = "\n".join([f"{op}. {ans}" for op, ans in options.items()])
    
    # Complete question including options
    question = base_question + "\n" + options_str
    
    # Original answer format
    answer = str(explanation) + '\n#### ' + answer_letter
    
    return {
        'question': question,
        'answer': answer,
        'answer_string': answer_string,
        'answer_letter': answer_letter,
        'answer_idx': answer_idx,
        'options': options,
        'options_str': options_str
    }

def _process_mmlu_item(item, rng):
    """Process MMLU data items"""
    base_question = item['input']
    raw_options = [item['A'], item['B'], item['C'], item['D']]
    correct_answer = item['target']
    
    # Get correct answer string
    original_idx = ord(correct_answer) - ord('A')
    answer_string = raw_options[original_idx]
    
    # Randomly shuffle options
    options_list = raw_options.copy()
    rng.shuffle(options_list)
    
    # Find the new position of the correct answer
    answer_idx = options_list.index(answer_string)
    answer_letter = chr(ord('A') + answer_idx)
    
    # Create options dictionary and string
    options = {chr(ord('A') + i): opt for i, opt in enumerate(options_list)}
    options_str = "\n".join([f"{op}. {ans}" for op, ans in options.items()])
    
    # Complete question including options
    question = base_question + "\n" + options_str
    
    # Original answer format
    answer = answer_letter
    
    return {
        'question': question,
        'answer': answer,
        'answer_string': answer_string,
        'answer_letter': answer_letter,
        'answer_idx': answer_idx,
        'options': options,
        'options_str': options_str
    }

def _process_medqa_item(item, rng):
    """Process MedQA data items"""
    base_question = item['question']
    choices = item['choices']
    ground_truth = item['answer'][0]
    
    # Get correct answer string
    answer_string = ground_truth
    
    # Randomly shuffle options
    options_list = choices.copy()
    rng.shuffle(options_list)
    
    # Find the new position of the correct answer
    answer_idx = options_list.index(answer_string)
    answer_letter = chr(ord('A') + answer_idx)
    
    # Create options dictionary and string
    options = {chr(ord('A') + i): opt for i, opt in enumerate(options_list)}
    options_str = "\n".join([f"{op}. {ans}" for op, ans in options.items()])
    
    # Complete question including options
    question = base_question + "\n" + options_str
    
    # Original answer format
    answer = answer_letter
    
    return {
        'question': question,
        'answer': answer,
        'answer_string': answer_string,
        'answer_letter': answer_letter,
        'answer_idx': answer_idx,
        'options': options,
        'options_str': options_str
    }

def _process_bioasq_item(item, rng):
    """Process BioASQ data items, converting to binary choice question format"""
    base_question = item['body']
    correct_answer = item['exact_answer']
    
    # Build context
    contexts = 'Context:\n'
    for snippet in item['snippets']:
        contexts += snippet['text'] + '\n'
    
    full_base_question = contexts + base_question
    
    # Define two options
    raw_options = ['yes', 'no']
    answer_string = correct_answer.lower()
    
    # Randomly shuffle options
    options_list = raw_options.copy()
    rng.shuffle(options_list)
    
    # Find the new position of the correct answer
    answer_idx = options_list.index(answer_string)
    answer_letter = chr(ord('A') + answer_idx)
    
    # Create options dictionary and string
    options = {chr(ord('A') + i): opt for i, opt in enumerate(options_list)}
    options_str = "\n".join([f"{op}. {ans}" for op, ans in options.items()])
    
    # Complete question including options
    question = full_base_question + "\n" + options_str
    
    # Original answer format
    answer = correct_answer
    
    return {
        'question': question,
        'answer': answer,
        'answer_string': answer_string,
        'answer_letter': answer_letter,
        'answer_idx': answer_idx,
        'options': options,
        'options_str': options_str
    }
