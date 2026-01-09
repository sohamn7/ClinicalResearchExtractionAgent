import pandas as pd
from datasets import load_dataset
from agent.doc_retrieval import DocRetrieval
from agent.training import load_environment, validate_environment
import os
import itertools
import ast
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from Bio import Entrez

# --- CONFIGURATION FOR ANALYSIS ---
MODEL_NAME = 'all-MiniLM-L6-v2'

# =====================================================================================
# ANALYSIS FUNCTIONS (Previously in analyze_retrieval.py)
# =====================================================================================

def safe_literal_eval(val):
    """
    Safely evaluate a string that should be a Python literal (e.g., a list of dicts).
    """
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError, MemoryError):
        return [] # Return empty list if parsing fails

def fetch_abstracts(ids, email):
    """
    Fetch abstracts for a list of PubMed or PMC IDs.
    """
    Entrez.email = email
    if not ids:
        return {}
    
    valid_ids = [str(i) for i in ids if i and isinstance(i, (str, int, float)) and not pd.isna(i)]
    if not valid_ids:
        return {}

    try:
        handle = Entrez.efetch(db="pubmed", id=valid_ids, rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        
        abstracts = {}
        for record in records.get('PubmedArticle', []):
            pmid = record['MedlineCitation']['PMID']
            article = record['MedlineCitation']['Article']
            abstract_text = ""
            if 'Abstract' in article:
                abstract_text = ' '.join(article['Abstract']['AbstractText'])
            abstracts[str(pmid)] = abstract_text
        return abstracts
    except Exception as e:
        print(f"Warning: Entrez fetch failed for IDs {valid_ids}. Error: {e}")
        return {id_str: "" for id_str in valid_ids}

def analyze_retrieval_performance(agent_df, test_df):
    """
    Performs a high-level analysis by comparing the aggregated topic embedding
    of retrieved articles vs. gold-standard articles for each prompt.
    Accepts dataframes directly instead of reading from files.
    """
    print("\n--- Starting Performance Analysis ---")
    
    # --- 1. Initialize Model ---
    print(f"Loading sentence transformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # --- 2. Process Each Prompt ---
    # Create a consistent string representation for matching, as direct dict comparison is tricky
    agent_df['pico_str'] = agent_df['prompt_pico'].astype(str)
    test_df['pico_str'] = test_df['pico'].astype(str)
    
    prompts = test_df['pico_str'].unique()
    all_prompt_metrics = []
    
    email = os.getenv('PUBMED_EMAIL')
    if not email:
        print("Error: PUBMED_EMAIL environment variable not set for analysis. Cannot fetch abstracts.")
        return

    print("Analyzing prompts using Cosine Similarity...")
    for prompt in tqdm(prompts, desc="Processing Prompts"):
        agent_articles = agent_df[agent_df['pico_str'] == prompt]
        test_row = test_df[test_df['pico_str'] == prompt].iloc[0]
        
        # The data is now in memory, so we don't need to parse it from a string
        gold_pubs_raw = test_row['screened_publication']
        gold_articles = pd.DataFrame(gold_pubs_raw if gold_pubs_raw else [])

        if gold_articles.empty or agent_articles.empty or agent_articles['retrieved_pmcid'].isnull().all():
            all_prompt_metrics.append({"prompt": prompt, "cosine_similarity": 0.0})
            continue

        # --- 3. Fetch Abstracts ---
        gold_pmids = gold_articles['pmid'].dropna().astype(str).tolist()
        agent_pmcids = agent_articles['retrieved_pmcid'].dropna().astype(str).tolist()
        
        all_ids = list(set(gold_pmids + agent_pmcids))
        abstracts = fetch_abstracts(all_ids, email)

        # --- 4. Aggregate Text and Create Single Embedding ---
        def get_full_text(df, id_col, title_col):
            texts = []
            for _, row in df.iterrows():
                doc_id = str(row[id_col])
                title = row[title_col]
                abstract = abstracts.get(doc_id, "")
                if title and isinstance(title, str):
                    texts.append(f"{title}\n\n{abstract}")
            return "\n\n---\n\n".join(texts)

        agent_full_text = get_full_text(agent_articles, 'retrieved_pmcid', 'retrieved_title')
        gold_full_text = get_full_text(gold_articles, 'pmid', 'title')

        if not agent_full_text or not gold_full_text:
            all_prompt_metrics.append({"prompt": prompt, "cosine_similarity": 0.0})
            continue

        agent_embedding = model.encode(agent_full_text, convert_to_tensor=True)
        gold_embedding = model.encode(gold_full_text, convert_to_tensor=True)

        # --- 5. Calculate Cosine Similarity ---
        cos_sim = util.cos_sim(agent_embedding, gold_embedding).item()
        
        all_prompt_metrics.append({
            "prompt": prompt,
            "cosine_similarity": cos_sim
        })

    # --- 6. Final Report ---
    if not all_prompt_metrics:
        print("\nNo metrics were calculated.")
        return
        
    report_df = pd.DataFrame(all_prompt_metrics)
    
    print("\n--- Retrieval Performance Analysis (Cosine Similarity) ---")
    print("\n--- Per-Prompt Similarity Score ---")
    pd.set_option('display.max_colwidth', None)
    print(report_df.to_string(index=False, float_format="%.4f"))
    
    print("\n--- Overall Average Metrics ---")
    avg_similarity = report_df['cosine_similarity'].mean()
    
    print(f"Average Cosine Similarity: {avg_similarity:.4f}")
    print("\nAnalysis complete.")

# =====================================================================================
# EVALUATION FUNCTIONS (Previously in doc_retrieval_testing.py)
# =====================================================================================

def get_extraction_train_df():
    """
    Loads the training data for extraction, downloading a small sample if not found locally.
    This data is used to create a sample for the DocRetrieval agent.
    """
    filename = 'extraction_train.jsonl'
    if os.path.exists(filename):
        print(f"Found cached '{filename}'. Loading from disk.")
        return pd.read_json(filename, lines=True)
    else:
        print(f"'{filename}' not found. Downloading a small sample from Hugging Face.")
        try:
            dataset = load_dataset("zifeng-ai/LEADSInstruct", "result_extraction", split="train", streaming=True)
            sample_data = list(dataset.take(500))
            df = pd.DataFrame(sample_data)
            
            stringify_cols = ['denomValue', 'numValue', 'pmid']
            for col in stringify_cols:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: str(x) if x is not None else None)
            
            df.to_json(filename, orient='records', lines=True)
            print(f"Successfully created and cached a sample in '{filename}'.")
            return df
        except Exception as e:
            print(f"Failed to download or process dataset sample: {e}")
            return None

def run_doc_retrieval_evaluation():
    """
    Main function to run the document retrieval evaluation and analysis workflow.
    """
    print("Loading environment variables...")
    load_environment()
    if not validate_environment():
        print("Exiting due to missing environment variables.")
        return

    print("Loading datasets...")
    try:
        test_dataset = load_dataset("zifeng-ai/LEADSInstruct", "study_screening", split="test", streaming=True)
        test_df = pd.DataFrame(list(test_dataset))
    except Exception as e:
        print(f"Failed to load study_screening dataset from Hugging Face: {e}")
        return

    extraction_train_df = get_extraction_train_df()
    if extraction_train_df is None:
        print("Could not load extraction training data. Aborting.")
        return

    sample_df = extraction_train_df.sample(n=15, random_state=42)
    sample_csv_path = 'temp_sample_df_for_doc_retrieval.csv'
    sample_df.to_csv(sample_csv_path, index=False)

    test_df_sample = test_df.sample(n=10, random_state=42)

    print(f"Starting document retrieval evaluation on {len(test_df_sample)} test prompts...")

    all_retrieved_articles = []

    for index, row in test_df_sample.iterrows():
        prompt = row['pico']
        print(f"\n{'='*20}\nPROMPT: {prompt}\n{'='*20}")

        doc_retriever = DocRetrieval(file=sample_csv_path, prompt=str(prompt))
        retrieved_articles_map = doc_retriever.doc_retrieval()

        if not isinstance(retrieved_articles_map, dict):
            retrieved_articles_map = {}

        retrieved_articles = dict(itertools.islice(retrieved_articles_map.items(), 10))

        print(f"Agent Retrieved {len(retrieved_articles)} articles:")
        if not retrieved_articles:
            print("  -> No articles were retrieved.")
            all_retrieved_articles.append({
                'prompt_pico': prompt, # Store the raw dict
                'retrieved_pmcid': None,
                'retrieved_title': 'No articles retrieved'
            })
        else:
            for i, article_data in enumerate(retrieved_articles.values()):
                print(f"  {i+1}. {article_data.get('title', 'No Title Found')} (PMCID: {article_data.get('pmc_id', 'N/A')})")
                all_retrieved_articles.append({
                    'prompt_pico': prompt, # Store the raw dict
                    'retrieved_pmcid': article_data.get('pmc_id'),
                    'retrieved_title': article_data.get('title')
                })

        ground_truth_pubs = row.get('screened_publication', [])
        print(f"\nGround Truth has {len(ground_truth_pubs)} articles.")

    agent_output_df = pd.DataFrame(all_retrieved_articles)
    agent_output_filename = 'doc_retrieval_agent_output.csv'
    # For CSV saving, we must stringify the dict column
    agent_output_df_to_save = agent_output_df.copy()
    agent_output_df_to_save['prompt_pico'] = agent_output_df_to_save['prompt_pico'].astype(str)
    agent_output_df_to_save.to_csv(agent_output_filename, index=False)
    print(f"\nAgent's retrieved articles saved to '{agent_output_filename}'")

    testing_df_filename = 'doc_retrieval_test_set_sample.csv'
    test_df_sample.to_csv(testing_df_filename, index=False)
    print(f"Document retrieval test set sample saved to '{testing_df_filename}'")

    os.remove(sample_csv_path)
    print(f"\n--- Retrieval Evaluation Finished ---")

    # --- CALL ANALYSIS FUNCTION ---
    analyze_retrieval_performance(agent_output_df, test_df_sample)


if __name__ == "__main__":
    run_doc_retrieval_evaluation()