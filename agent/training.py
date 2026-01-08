import os
import pandas as pd
import json
from .doc_retrieval import DocRetrieval
from .context_extract import Extract
from .verification import Verify

def load_environment():
    """Load environment variables from .env file if it exists."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ Environment variables loaded from .env file")
    except ImportError:
        print("⚠️  python-dotenv not installed. Make sure to set environment variables manually.")
    except Exception as e:
        print(f"⚠️  Could not load .env file: {e}")

def validate_environment():
    """Validate that required environment variables are set."""
    required_vars = [
        'ENDPOINT_URL',
        'AZURE_OPENAI_API_KEY',
        'DEPLOYMENT_NAME',
        'ENTREZ_API_KEY',
        'PUBMED_EMAIL'
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        return False
    print("✓ All required environment variables are set")
    return True

def load_and_cache_data(jsonl_path="LEADSInstructtrain.jsonl", cache_path="dataframe_cache.pkl"):
    """Loads dataframe from cache if exists, otherwise from jsonl and caches it."""
    if os.path.exists(cache_path):
        print("Loading DataFrame from cache...")
        return pd.read_pickle(cache_path)
    else:
        print("Loading DataFrame from JSONL and caching it...")
        df = pd.read_json(jsonl_path, lines=True)
        # Create a sample dataframe
        train_df = df.sample(frac=0.8, random_state=42)
        sample_df = train_df.sample(frac=0.0003, random_state=42)
        sample_df.to_pickle(cache_path)
        return sample_df

def run_workflow(sample_df):
    """
    Runs the full document retrieval and data extraction workflow.
    """
    temp_csv_path = "temp_sample_for_training.csv"
    sample_df.to_csv(temp_csv_path, index=False)
    
    prompt = "Extract clinical trial data based on the provided columns."

    try:
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING WORKFLOW")
        print("="*60)

        # 1. Document Retrieval
        print("\n[Step 1/3] Retrieving documents...")
        doc_retriever = DocRetrieval(temp_csv_path, prompt)
        doc_retriever.sample_df = sample_df

        doc_retriever.info_map = {}
        for idx, row in sample_df.iterrows():
            pmc_id = row.get('pmc_id', f'synthetic_pmc_id_{idx}')
            title = row.get('title', f'Synthetic Title {idx}')
            authors_raw = row.get('authors', '[]')
            try:
                authors = json.loads(authors_raw) if isinstance(authors_raw, str) else authors_raw
            except json.JSONDecodeError:
                authors = [authors_raw] if isinstance(authors_raw, str) else []
            paper_content = row.get('paper_content', '')

            doc_retriever.info_map[idx] = {
                "pmc_id": pmc_id,
                "title": title,
                "authors": authors,
                "text": paper_content
            }
        
        if not doc_retriever.info_map:
            print("❌ doc_retriever.info_map is empty after parsing sample_df. Stopping workflow.")
            return None
        
        print(f"✓ Populated doc_retriever.info_map with {len(doc_retriever.info_map)} articles from sample_df.")

        # 2. Context Extraction
        print("\n[Step 2/3] Extracting data from documents...")
        extractor = Extract()
        # print(doc_retriever.info_map)
        extracted_df = extractor.context_extract(doc_retriever)

        if extracted_df is None or extracted_df.empty:
            print("❌ Initial data extraction resulted in an empty dataset.")
            return None

        print("✓ Initial data extraction complete.")
        
        # 3. Data Quality Verification
        print("\n[Step 3/3] Verifying and refining data quality...")
        verifier = Verify()
        max_retries = 3
        final_df = extracted_df

        for i in range(max_retries):
            print(f"\n--- Verification Attempt {i + 1}/{max_retries} ---")
            
            current_df, is_quality_sufficient = verifier.verify_data_quality(
                extracted_df=final_df,
                sample_df=doc_retriever.sample_df,
                context_extractor=extractor,
                doc_retrieval_instance=doc_retriever
            )
            final_df = current_df

            if is_quality_sufficient:
                print("✅ Data quality meets the threshold.")
                break
            else:
                if i < max_retries - 1:
                    print("⚠️ Data quality below threshold. Retrying verification with regenerated data.")
                else:
                    print("❌ Max retries reached. Proceeding with the last generated dataset.")
        
        print("✓ Verification and refinement step complete.")
        return final_df

    except Exception as e:
        print(f"❌ An error occurred during the workflow: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)

def main():
    """Main function to run the training workflow."""
    print("🚀 Starting Training Workflow")
    
    load_environment()
    
    if not validate_environment():
        print("❌ Environment validation failed. Please check your environment variables.")
        return
    
    sample_df = load_and_cache_data()
    
    final_dataframe = run_workflow(sample_df)

    if final_dataframe is not None:
        print("\n" + "="*60)
        print("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        output_filename = "leads_result.csv"
        final_dataframe.to_csv(output_filename, index=False)
        print(f"✓ Data saved to {output_filename}")

    else:
        print("\n" + "="*60)
        print("❌ WORKFLOW FAILED")
        print("="*60)
        print("Please check the error messages above.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Workflow interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()