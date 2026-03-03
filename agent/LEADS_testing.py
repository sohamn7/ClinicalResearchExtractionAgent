import os
import sys
import pandas as pd
import json
import re
from tqdm import tqdm
import numpy as np

# Add the project root to sys.path to allow absolute imports when run directly
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent.context_extract import Extract, clean_json_from_llm
from agent.LEADS_extract import LEADSExtract
from agent.training import load_environment, validate_environment
from bert_score import score as bert_scorer
import ast


def normalize_text(text):
    """Lowercase, remove punctuation, and strip whitespace."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'[^\w\s]', '', text.lower()).strip()


def compare_text_bert(gold, extracted, threshold=0.75):
    """
    Compares two strings using BERTScore F1.
    Returns True if the F1 score is above the threshold.
    """
    if not isinstance(gold, str) or not gold.strip() or not isinstance(extracted, str) or not extracted.strip():
        return False
    try:
        P, R, F1 = bert_scorer([extracted], [gold], lang='en', verbose=False)
        f1_score = F1.item()
        return f1_score >= threshold
    except Exception as e:
        print(f"Warning: BERTScore calculation failed with error: {e}. Defaulting to False.")
        return False


def extract_first_number(text):
    """Extracts the first floating point number from a string."""
    if isinstance(text, (int, float)):
        return float(text)
    if not isinstance(text, str):
        return None
    match = re.search(r'-?\d+\.?\d*|-?\.\d+', text)
    if match:
        try:
            return float(match.group(0))
        except (ValueError, TypeError):
            return None
    return None


def is_numeric_zero_or_na(value):
    """Checks if a value is considered zero, null, or N/A for numeric comparison."""
    if value is None:
        return True
    val_str = str(value).strip().lower()
    if val_str in ['n/a', 'nan', 'null', '']:
        return True
    num = extract_first_number(val_str)
    return num == 0.0


def compare_numeric(gold, extracted, tolerance=0.02):
    """
    Compare two values as numbers with a given tolerance after extracting numeric parts.
    Treats 0, N/A, and Null as equivalent.
    """
    if is_numeric_zero_or_na(gold) and is_numeric_zero_or_na(extracted):
        return True

    gold_num = extract_first_number(str(gold))
    extracted_num = extract_first_number(str(extracted))

    if gold_num is None or extracted_num is None:
        return normalize_text(str(gold)) == normalize_text(str(extracted))

    if gold_num == extracted_num:
        return True
    
    if gold_num == 0:
        return abs(extracted_num) <= tolerance

    return abs((gold_num - extracted_num) / gold_num) <= tolerance


def get_value_from_dict_string(s):
    """
    Safely parses a string that looks like a dictionary and extracts the 'value' key.
    """
    if not isinstance(s, str) or not s.startswith('{'):
        return s
    try:
        data = ast.literal_eval(s)
        if isinstance(data, dict):
            val = data.get('value')
            if val is not None:
                return val
            return s
    except (ValueError, SyntaxError):
        return s
    return s


def run_evaluation():
    """
    Main function to run the evaluation workflow.
    Uses the original Extract for numerical columns and LEADSExtract for semantic columns.
    """
    print("Loading dataset...")
    try:
        full_df = pd.read_csv('ADAMay12.csv')
    except FileNotFoundError:
        print("Error: ADAMay12.csv not found in the project root directory.")
        return

    # Take a small sample for testing
    test_df = full_df.sample(n=10).reset_index()

    # Get the indices of the rows used in test_df to exclude them from sample_df
    test_indices = test_df.index.tolist()

    # Create a temporary DataFrame excluding test_df rows
    remaining_df = full_df.drop(test_df.index)

    n_sample = min(10, len(remaining_df))
    if n_sample == 0:
        print("Error: Not enough data in ADAMay12.csv to create a distinct sample_df for context.")
        return

    sample_df = remaining_df.sample(n=n_sample).reset_index(drop=True)

    print(f"Created a sample dataframe of {len(sample_df)} rows for LLM context and a test dataframe of {len(test_df)} rows for evaluation.")

    # --- Column definitions ---
    numeric_columns = ['percentage', 'n_study', 'n_positive']
    semantic_columns = ['trial_name', 'patient_characteristics', 'dosage', 'method', 'notes']
    target_columns = numeric_columns + semantic_columns

    # --- Set up the original Extract for numerical columns ---
    extractor = Extract()
    extractor.user_prompt = "extract clinical data from studies for the anti-drug antibody (ADA) reactions from different therapeutic drugs in patients and patient groups"
    extractor.cols = numeric_columns

    # --- Build info_map for LEADS semantic extraction ---
    info_map = {}
    for idx, row in test_df.iterrows():
        context = row.get('context')
        if isinstance(context, str) and context.strip():
            info_map[idx] = {
                "pmc_id": str(row.get('index', idx)),
                "title": f"Test Article {idx}",
                "authors": [],
                "text": context
            }

    # Run LEADS extraction on all articles for semantic columns
    print("\n[Step 1/2] Running LEADS extraction for semantic columns...")
    leads_extractor = LEADSExtract(
        info_map=info_map,
        semantic_cols=semantic_columns
    )
    leads_output_df = leads_extractor.run_extraction()

    # Map LEADS results back by info_map key for lookup
    leads_by_idx = {}
    info_map_keys = list(info_map.keys())
    for i, key in enumerate(info_map_keys):
        if i < len(leads_output_df):
            leads_by_idx[key] = leads_output_df.iloc[i].to_dict()

    # --- Metrics ---
    user_metrics = {col: {'total_matches': 0, 'gold_present': 0, 'extracted_present': 0} for col in target_columns}

    papers_fully_correct = 0
    total_papers = 0
    all_extracted_data_for_csv = []

    print(f"\n[Step 2/2] Evaluating on {len(test_df)} test papers (LLM for numerical, LEADS for semantic)...")

    for index, gold_row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        total_papers += 1
        context = gold_row.get('context')

        current_extracted = {}

        row_for_csv = {'original_index': gold_row.get('index')}
        for col in target_columns:
            row_for_csv[f'{col}_gold'] = gold_row.get(col)
            row_for_csv[f'{col}_extracted'] = None

        # --- Numerical extraction via original Extract (LLM) ---
        if isinstance(context, str) and context.strip():
            new_data_json_raw = extractor.extract_data(context, sample_df[numeric_columns])

            try:
                cleaned_json = clean_json_from_llm(new_data_json_raw)
                parsed_llm_data = json.loads(cleaned_json)
                if isinstance(parsed_llm_data, list) and parsed_llm_data:
                    current_extracted.update(parsed_llm_data[0])
                elif isinstance(parsed_llm_data, dict):
                    current_extracted.update(parsed_llm_data)
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: Could not parse LLM output for a row: {new_data_json_raw}")

        # --- Semantic extraction via LEADS ---
        leads_result = leads_by_idx.get(index, {})
        current_extracted.update(leads_result)

        for col in target_columns:
            row_for_csv[f'{col}_extracted'] = current_extracted.get(col)

        all_extracted_data_for_csv.append(row_for_csv)

        # --- Comparison ---
        is_paper_fully_correct = True
        for col in target_columns:
            raw_gold = gold_row.get(col)
            raw_extracted = current_extracted.get(col)

            gold_value = get_value_from_dict_string(raw_gold)
            extracted_value = get_value_from_dict_string(raw_extracted)

            gold_present = gold_value is not None and str(gold_value).strip() != "" and str(gold_value).strip().lower() not in ['nan', 'n/a']
            extracted_present = extracted_value is not None and str(extracted_value).strip() != "" and str(extracted_value).strip().lower() not in ['nan', 'n/a']

            if gold_present:
                user_metrics[col]['gold_present'] += 1
            if extracted_present:
                user_metrics[col]['extracted_present'] += 1

            match = False
            if col in numeric_columns:
                match = compare_numeric(gold_value, extracted_value)
            elif gold_present and extracted_present:
                match = compare_text_bert(str(gold_value), str(extracted_value), threshold=0.8)
            elif not gold_present and not extracted_present:
                # Both are N/A — this is a match for Accuracy
                match = True

            if match:
                user_metrics[col]['total_matches'] += 1
            else:
                is_paper_fully_correct = False

        if is_paper_fully_correct:
            papers_fully_correct += 1

    extracted_results_df = pd.DataFrame(all_extracted_data_for_csv)
    extracted_results_df.to_csv('leads_test_results.csv', index=False)
    print("\nAgent's extracted results saved to 'leads_test_results.csv'")

    columns_to_save = ['index'] + target_columns
    test_df[columns_to_save].to_csv('leads_test_gold.csv', index=False)
    print("Test data (original index and target columns) saved to 'leads_test_gold.csv'")

    # --- Evaluation Report ---
    print("\n--- Evaluation Report ---")

    report_data = []
    agg_metrics = {
        'numerical': {'total_matches': 0, 'num_cols': 0, 'extracted_present': 0, 'gold_present': 0},
        'semantic': {'total_matches': 0, 'num_cols': 0, 'extracted_present': 0, 'gold_present': 0},
    }

    total_rows = len(test_df)

    for col, scores in user_metrics.items():
        total_matches = scores['total_matches']
        extracted = scores['extracted_present']
        gold = scores['gold_present']

        accuracy = total_matches / total_rows if total_rows > 0 else 0.0

        if gold > 0:
            recall = 1 - (abs(gold - extracted) / gold)
            recall = max(0, recall)
        else:
            recall = 1.0 if extracted == 0 else 0.0

        report_data.append({
            "Field": col,
            "Accuracy": f"{accuracy:.2%}",
            "Recall (by count)": f"{recall:.2%}",
            "Total Matches": total_matches,
            "Extracted": extracted,
            "Gold": gold
        })

        category = 'numerical' if col in numeric_columns else 'semantic'
        agg_metrics[category]['total_matches'] += total_matches
        agg_metrics[category]['num_cols'] += 1
        agg_metrics[category]['extracted_present'] += extracted
        agg_metrics[category]['gold_present'] += gold

    report_df = pd.DataFrame(report_data)
    print(report_df.to_string(index=False))

    # Aggregate Metrics
    print("\n--- Aggregate Metrics ---")
    for category in ['numerical', 'semantic']:
        scores = agg_metrics[category]
        total_matches = scores['total_matches']
        num_cols = scores['num_cols']
        extracted = scores['extracted_present']
        gold = scores['gold_present']

        total_cells_in_category = total_rows * num_cols
        accuracy = total_matches / total_cells_in_category if total_cells_in_category > 0 else 0.0

        if gold > 0:
            recall = 1 - (abs(gold - extracted) / gold)
            recall = max(0, recall)
        else:
            recall = 1.0 if extracted == 0 else 0.0

        print(f"\n--- {category.capitalize()} ---")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Recall (by count):    {recall:.2%}")

    print("\n--- Final Score ---")

    # Overall Accuracy and Recall
    total_cells = len(test_df) * len(target_columns)
    total_overall_matches = agg_metrics['numerical']['total_matches'] + agg_metrics['semantic']['total_matches']
    overall_accuracy = total_overall_matches / total_cells if total_cells > 0 else 0

    total_gold = agg_metrics['numerical']['gold_present'] + agg_metrics['semantic']['gold_present']
    total_extracted = agg_metrics['numerical']['extracted_present'] + agg_metrics['semantic']['extracted_present']

    if total_gold > 0:
        overall_recall = 1 - (abs(total_gold - total_extracted) / total_gold)
        overall_recall = max(0, overall_recall)
    else:
        overall_recall = 1.0 if total_extracted == 0 else 0.0

    print(f"\n--- Overall (Aggregated) ---")
    print(f"  Accuracy (Matches / Total Cells): {overall_accuracy:.2%}")
    print(f"  Recall (by count):                {overall_recall:.2%}")

    all_correct_rate = papers_fully_correct / total_papers if total_papers > 0 else 0
    print(f"\nPer-paper 'all fields correct' rate: {all_correct_rate:.2%} ({papers_fully_correct}/{total_papers} papers)")



if __name__ == "__main__":
    load_environment()
    
    if not validate_environment():
        print("Exiting due to missing environment variables.")
        exit(1)

    run_evaluation()
