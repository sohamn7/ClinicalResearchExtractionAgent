import pandas as pd
import json
import re
from tqdm import tqdm
import numpy as np
from .context_extract import Extract, clean_json_from_llm
from .training import load_environment, validate_environment
from bert_score import score as bert_scorer # Import BERTScore

def normalize_text(text):
    """Lowercase, remove punctuation, and strip whitespace."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'[^\w\s]', '', text.lower()).strip()

def compare_text_bert(gold, extracted, threshold=0.8):
    """
    Compares two strings using BERTScore F1.
    Returns True if the F1 score is above the threshold.
    """
    # Ensure inputs are non-empty strings, otherwise similarity is meaningless
    if not isinstance(gold, str) or not gold.strip() or not isinstance(extracted, str) or not extracted.strip():
        return False

    # bert_scorer expects lists of strings (candidates, references)
    # It returns (P, R, F1) as tensors. We are interested in F1.
    try:
        P, R, F1 = bert_scorer([extracted], [gold], lang='en', verbose=False)
        f1_score = F1.item()
        return f1_score >= threshold
    except Exception as e:
        print(f"Warning: BERTScore calculation failed with error: {e}. Defaulting to False.")
        return False

def compare_numeric(gold, extracted, tolerance=0.01):
    """
    Compare two values as numbers with a given tolerance.
    Returns True if they match, False otherwise.
    """
    try:
        gold_num = float(gold)
        extracted_num = float(extracted)
        if gold_num == 0:
            return extracted_num == 0
        return abs((gold_num - extracted_num) / gold_num) <= tolerance
    except (ValueError, TypeError):
        # If conversion to float fails, fall back to normalized text match
        return normalize_text(str(gold)) == normalize_text(str(extracted))

def parse_results_json(json_str):
    """
    Parses the 'results' JSON string into a dictionary of {title: value}.
    """
    if not isinstance(json_str, str):
        return {}
    try:
        data = json.loads(json_str)
        if not isinstance(data, list):
            return {}
        results_map = {normalize_text(item.get("title", "")): item.get("value") for item in data}
        return results_map
    except (json.JSONDecodeError, TypeError):
        return {}

def compare_results(gold_json_str, extracted_json_str):
    """
    Compares the 'results' field by parsing the JSON and comparing the contents.
    """
    gold_map = parse_results_json(gold_json_str)
    extracted_map = parse_results_json(extracted_json_str)

    if not gold_map and not extracted_map:
        return True
    
    if set(gold_map.keys()) != set(extracted_map.keys()):
        return False

    for title, gold_value in gold_map.items():
        extracted_value = extracted_map.get(title)
        if not compare_numeric(gold_value, extracted_value):
            return False
            
    return True

def run_evaluation():
    """
    Main function to run the evaluation workflow.
    """
    print("Loading datasets...")
    try:
        test_df = pd.read_json('extraction_test.jsonl', lines=True)
        train_df = pd.read_json('extraction_train.jsonl', lines=True)
    except FileNotFoundError as e:
        print(f"Error: Make sure '{e.filename}' is in the project root directory.")
        return

    test_df = test_df.sample(n=10, random_state=42).reset_index(drop=True)

    print("Creating a 15-row sample from the training data for context...")
    sample_df_full = train_df.sample(n=15, random_state=42)
    sample_df = sample_df_full.drop(columns=['paper_content'], errors='ignore')

    extractor = Extract()
    target_columns = [
        'outcomeDef', 'groupDef', 'paramType', 'unitOfMeasure', 
        'timeFrame', 'unitOfDenom', 'denomValue', 'results'
    ]
    extractor.cols = target_columns

    field_metrics = {col: {'tp': 0, 'fp': 0, 'fn': 0} for col in target_columns}
    papers_fully_correct = 0
    total_papers = 0
    all_extracted_data = []

    print(f"Starting evaluation on {len(test_df)} test papers...")
    
    for index, gold_row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        total_papers += 1
        paper_content = gold_row.get('paper_content')
        
        extracted_data = {
            'id': gold_row.get('id'), 'pmid': gold_row.get('pmid'), 'nctid': gold_row.get('nctid')
        }

        if not isinstance(paper_content, str) or not paper_content.strip():
            all_extracted_data.append(extracted_data) 
            continue

        extracted_results_list = extractor.process_article(
            article={'text': paper_content}, 
            df=sample_df
        )
        
        if extracted_results_list:
            extracted_data.update(extracted_results_list[0])
        
        all_extracted_data.append(extracted_data)

        is_paper_fully_correct = True
        for col in target_columns:
            gold_value = gold_row.get(col)
            extracted_value = extracted_data.get(col)

            gold_present = gold_value is not None and str(gold_value).strip() != "" and gold_value != []
            extracted_present = extracted_value is not None and str(extracted_value).strip() != "" and extracted_value != []

            match = False
            if gold_present and extracted_present:
                numeric_cols = ['denomValue']
                results_cols = ['results']
                
                if col in results_cols:
                    match = compare_results(gold_value, extracted_value)
                elif col in numeric_cols:
                    match = compare_numeric(gold_value, extracted_value)
                else: # Default to BERTScore for all other text fields
                    match = compare_text_bert(str(gold_value), str(extracted_value), threshold=0.7)

            elif not gold_present and not extracted_present:
                continue

            if match:
                field_metrics[col]['tp'] += 1
            else:
                is_paper_fully_correct = False
                if extracted_present and not gold_present:
                    field_metrics[col]['fp'] += 1
                elif not extracted_present and gold_present:
                    field_metrics[col]['fn'] += 1
                elif extracted_present and gold_present:
                    field_metrics[col]['fp'] += 1
                    field_metrics[col]['fn'] += 1
        
        if is_paper_fully_correct:
            papers_fully_correct += 1

    extracted_results_df = pd.DataFrame(all_extracted_data)
    extracted_results_df.to_csv('extract_results.csv', index=False)
    print("\nAgent's extracted results saved to 'extract_results.csv'")

    test_df_to_save = test_df.drop(columns=['paper_content'], errors='ignore')
    test_df_to_save.to_csv('extraction_test_results.csv', index=False)
    print("Test data (gold standard) saved to 'extraction_test_results.csv' (without paper_content)")

    print("\n--- Evaluation Report ---")
    
    report_data = []
    for col, scores in field_metrics.items():
        tp, fp, fn = scores['tp'], scores['fp'], scores['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        report_data.append({
            "Field": col,
            "Precision": f"{precision:.2%}",
            "Recall": f"{recall:.2%}",
            "F1-Score": f"{f1:.2f}",
            "TP": tp,
            "FP": fp,
            "FN": fn
        })

    report_df = pd.DataFrame(report_data)
    print(report_df.to_string(index=False))

    print("\n--- Aggregate Metrics ---")
    all_correct_rate = papers_fully_correct / total_papers if total_papers > 0 else 0
    print(f"Per-paper 'all fields correct' rate: {all_correct_rate:.2%} ({papers_fully_correct}/{total_papers} papers)")


if __name__ == "__main__":
    load_environment()
    
    if not validate_environment():
        print("Exiting due to missing environment variables.")
        exit(1)

    run_evaluation()
