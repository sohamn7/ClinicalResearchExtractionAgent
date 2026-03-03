import os
import sys
import pandas as pd
import json
import re
from tqdm import tqdm
import numpy as np

# Add the project root to sys.path to allow absolute imports when run directly
# This assumes the script is run from the 'agent' directory or one level above it
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent.context_extract import Extract, clean_json_from_llm
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


def _extract_nct_numbers(text):
    """Extract all NCT/EudraCT identifiers from a string."""
    if not isinstance(text, str):
        return set()
    nct = set(re.findall(r'NCT\d+', text, re.IGNORECASE))
    eudract = set(re.findall(r'EudraCT\s*\d{4}-\d{6}-\d{2}', text, re.IGNORECASE))
    return nct | eudract


def _is_trial_identifier(text):
    """Check if a string looks like it's a valid trial name or identifier (not N/A)."""
    if not isinstance(text, str):
        return False
    clean = text.strip().lower()
    return clean not in ['', 'n/a', 'nan', 'null', 'none'] and len(clean) > 2


def compare_trial_name(gold, extracted):
    """
    Compare trial names with special handling for NCT numbers vs. trial titles.
    Both are valid trial identifiers — if gold is a title and extracted is an NCT (or vice versa),
    they should be treated as a match since both correctly identify the trial.
    """
    gold_str = str(gold).strip() if gold is not None else ""
    ext_str = str(extracted).strip() if extracted is not None else ""

    gold_valid = _is_trial_identifier(gold_str)
    ext_valid = _is_trial_identifier(ext_str)

    # If neither is present, it's a match (both N/A)
    if not gold_valid and not ext_valid:
        return True
    # If only one is present, it's not a match
    if not gold_valid or not ext_valid:
        return False

    # Check substring containment (e.g., "GATTO study" vs "GATTO study (NCT03360734)")
    if gold_str.lower() in ext_str.lower() or ext_str.lower() in gold_str.lower():
        return True

    # Check if both share the same NCT number
    gold_ncts = _extract_nct_numbers(gold_str)
    ext_ncts = _extract_nct_numbers(ext_str)
    if gold_ncts and ext_ncts and gold_ncts & ext_ncts:
        return True

    # If both have NCT/registry numbers but none overlap, they identify different trials
    if gold_ncts and ext_ncts:
        return False

    # If one has an NCT number and the other is a trial name/title,
    # both are valid trial identifiers — treat as match
    gold_has_nct = bool(gold_ncts)
    ext_has_nct = bool(ext_ncts)
    if gold_has_nct != ext_has_nct:
        # One is an NCT, the other is a title — both are correct identifiers
        return True

    # Both are titles — use BERTScore with a lower threshold (0.65)
    return compare_text_bert(gold_str, ext_str, threshold=0.65)


def _extract_dose_components(text):
    """
    Extract numeric dose values and frequency from a dosage string.
    Returns (dose_numbers, per_week_dose) where per_week_dose is an attempt
    to normalize to a weekly equivalent.
    """
    if not isinstance(text, str):
        return [], None

    # Extract all numbers
    numbers = [float(m) for m in re.findall(r'\d+\.?\d*', text)]

    # Try to compute per-week equivalent
    text_lower = text.lower()
    main_dose = numbers[0] if numbers else None
    per_week = None

    if main_dose is not None:
        if 'every 4 weeks' in text_lower or 'every four weeks' in text_lower or 'q4w' in text_lower:
            per_week = main_dose / 4.0
        elif 'every 3 weeks' in text_lower or 'every three weeks' in text_lower or 'q3w' in text_lower:
            per_week = main_dose / 3.0
        elif 'every 2 weeks' in text_lower or 'every two weeks' in text_lower or 'every other week' in text_lower or 'q2w' in text_lower:
            per_week = main_dose / 2.0
        elif 'every week' in text_lower or 'weekly' in text_lower or 'per week' in text_lower or '/week' in text_lower or 'q1w' in text_lower:
            per_week = main_dose
        elif 'every 21 days' in text_lower:
            per_week = main_dose / 3.0
        elif 'every 14 days' in text_lower:
            per_week = main_dose / 2.0
        elif 'every 28 days' in text_lower:
            per_week = main_dose / 4.0

    return numbers, per_week


def compare_dosage(gold, extracted):
    """
    Compare dosage values with unit-aware normalization.
    Handles cases like '8 mg/kg every 4 weeks' vs '2.0 mg/kg/week'.
    """
    gold_str = str(gold).strip() if gold is not None else ""
    ext_str = str(extracted).strip() if extracted is not None else ""

    gold_valid = gold_str.lower() not in ['', 'n/a', 'nan', 'null', 'none']
    ext_valid = ext_str.lower() not in ['', 'n/a', 'nan', 'null', 'none']

    if not gold_valid and not ext_valid:
        return True
    if not gold_valid or not ext_valid:
        return False

    # Try per-week normalization first
    gold_nums, gold_weekly = _extract_dose_components(gold_str)
    ext_nums, ext_weekly = _extract_dose_components(ext_str)

    if gold_weekly is not None and ext_weekly is not None:
        # Compare normalized per-week doses within 5% tolerance
        if gold_weekly == 0 and ext_weekly == 0:
            return True
        if gold_weekly > 0 and abs((gold_weekly - ext_weekly) / gold_weekly) <= 0.05:
            return True

    # Check if the primary dose numbers match (e.g., both mention "40 mg")
    if gold_nums and ext_nums:
        gold_set = set(gold_nums)
        ext_set = set(ext_nums)
        if gold_set & ext_set:  # Any shared number
            return True

    # Fallback to BERTScore with lower threshold for dosage text
    return compare_text_bert(gold_str, ext_str, threshold=0.65)

def extract_first_number(text):
    """Extracts the first floating point number from a string."""
    if isinstance(text, (int, float)):
        return float(text)
    if not isinstance(text, str):
        return None
    # This regex finds integers and floats, including those starting with a dot.
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
    # Use extract_first_number to see if the numeric value is 0
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

    # If either number couldn't be extracted, we can't compare them numerically.
    if gold_num is None or extracted_num is None:
        # Fallback for when numbers can't be extracted but strings are identical non-numerically
        return normalize_text(str(gold)) == normalize_text(str(extracted))

    if gold_num == extracted_num:
        return True
    
    # Avoid division by zero
    if gold_num == 0:
        # If gold is 0, extracted should be very close to 0 (within absolute tolerance)
        return abs(extracted_num) <= tolerance

    return abs((gold_num - extracted_num) / gold_num) <= tolerance

def get_value_from_dict_string(s):
    """
    Safely parses a string that looks like a dictionary and extracts the 'value' key.
    """
    if not isinstance(s, str) or not s.startswith('{'):
        return s  # Return as is if not a dict string
    try:
        data = ast.literal_eval(s)
        if isinstance(data, dict):
            val = data.get('value')
            if val is not None:
                return val
            # If 'value' is not there, return the original string so it's counted as present
            return s
    except (ValueError, SyntaxError):
        return s  # Return original string if parsing fails
    return s

def run_evaluation():
    """
    Main function to run the evaluation workflow.
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

    # If there are not enough rows in remaining_df to create a sample_df size 10,
    # sample what's available or adjust strategy. For now, take what's available.
    n_sample = min(10, len(remaining_df))
    if n_sample == 0:
        print("Error: Not enough data in ADAMay12.csv to create a distinct sample_df for context.")
        return

    sample_df = remaining_df.sample(n=n_sample).reset_index(drop=True)

    print(f"Created a sample dataframe of {len(sample_df)} rows for LLM context and a test dataframe of {len(test_df)} rows for evaluation.")

    extractor = Extract()
    extractor.user_prompt = "extract clinical data from studies for the anti-drug antibody (ADA) reactions from different therapeutic drugs in patients and patient groups"
    target_columns = [
        'percentage',
        'n_study',
        'n_positive',
        'trial_name',
        'patient_characteristics',
        'dosage',
        'method',
        'notes'
    ]
    extractor.cols = target_columns

    # Simplified metrics, focused on the user's definitions
    user_metrics = {col: {'total_matches': 0, 'gold_present': 0, 'extracted_present': 0} for col in target_columns}

    papers_fully_correct = 0
    total_papers = 0
    all_extracted_data_for_csv = []

    print(f"Starting evaluation on {len(test_df)} test papers...")

    for index, gold_row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        total_papers += 1
        context = gold_row.get('context')

        current_extracted_llm_output = {}

        row_for_csv = {'original_index': gold_row.get('index')}
        for col in target_columns:
            row_for_csv[f'{col}_gold'] = gold_row.get(col)
            row_for_csv[f'{col}_extracted'] = None

        if isinstance(context, str) and context.strip():
            new_data_json_raw = extractor.extract_data(context, sample_df[target_columns])

            try:
                cleaned_json = clean_json_from_llm(new_data_json_raw)
                parsed_llm_data = json.loads(cleaned_json)
                if isinstance(parsed_llm_data, list) and parsed_llm_data:
                    current_extracted_llm_output.update(parsed_llm_data[0])
                elif isinstance(parsed_llm_data, dict):
                    current_extracted_llm_output.update(parsed_llm_data)
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: Could not parse LLM output for a row: {new_data_json_raw}")

        for col in target_columns:
            raw_extracted = current_extracted_llm_output.get(col)
            row_for_csv[f'{col}_extracted'] = raw_extracted

        all_extracted_data_for_csv.append(row_for_csv)

        numeric_columns = ['percentage', 'n_study', 'n_positive']
        is_paper_fully_correct = True
        for col in target_columns:
            raw_gold = gold_row.get(col)
            raw_extracted = current_extracted_llm_output.get(col)

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
            elif col == 'trial_name':
                match = compare_trial_name(gold_value, extracted_value)
            elif col == 'dosage':
                match = compare_dosage(gold_value, extracted_value)
            elif col == 'patient_characteristics':
                if gold_present and extracted_present:
                    match = compare_text_bert(str(gold_value), str(extracted_value), threshold=0.7)
                elif not gold_present and not extracted_present:
                    match = True
            elif gold_present and extracted_present:
                match = compare_text_bert(str(gold_value), str(extracted_value), threshold=0.8)
            elif not gold_present and not extracted_present:
                # Both are N/A - this is a match for Accuracy
                match = True

            if match:
                user_metrics[col]['total_matches'] += 1
            else:
                is_paper_fully_correct = False

        if is_paper_fully_correct:
            papers_fully_correct += 1

    extracted_results_df = pd.DataFrame(all_extracted_data_for_csv)
    extracted_results_df.to_csv('extract_two_results.csv', index=False)
    print("\nAgent's extracted results saved to 'extract_two_results.csv'")

    columns_to_save = ['index'] + target_columns
    test_df[columns_to_save].to_csv('extract_two_test.csv', index=False)
    print("Test data (original index and target columns) saved to 'extract_two_test.csv'")

    # --- Simplified Evaluation Report ---
    print("\n--- Evaluation Report ---")

    report_data = []
    numeric_columns = ['percentage', 'n_study', 'n_positive']
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