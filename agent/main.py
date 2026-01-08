"""
Main script to test document retrieval functionality.
Allows users to input prompts and upload CSV files for clinical trial data extraction.
"""

import os
import sys
from pathlib import Path
from typing import Any
from .doc_retrieval import DocRetrieval
from .context_extract import Extract
from .verification import Verify
from .llm_util import LLMClient
import json
import pandas as pd

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
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        return False
    
    print("✓ All required environment variables are set")
    return True

def get_user_input():
    """Get user input for prompt and file path."""
    print("\n" + "="*60)
    print("CLINICAL TRIAL DOCUMENT RETRIEVAL SYSTEM")
    print("="*60)
    
    # Get user prompt
    print("\nEnter your clinical trial data requirements:")
    print("(e.g., 'Find studies on diabetes treatment with metformin')")
    prompt = input("Prompt: ").strip()
    
    if not prompt:
        print("❌ Prompt cannot be empty!")
        return None, None
    
    # Get file path
    print("\nEnter the path to your CSV file:")
    print("(e.g., 'data/clinical_trials.csv' or 'sample_data.csv')")
    file_path = input("File path: ").strip()
    
    if not file_path:
        print("❌ File path cannot be empty!")
        return None, None

    file_path = file_path
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None, None
    
    # Check if it's a CSV file
    if not file_path.lower().endswith('.csv'):
        print("⚠️  Warning: File doesn't have .csv extension")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return None, None
    
    return prompt, file_path

def test_llm_connection():
    """Test if LLM client can be initialized."""
    try:
        client = LLMClient()
        if hasattr(client, 'client'):
            print("✓ LLM client initialized successfully")
            return True
        else:
            print("❌ LLM client initialization failed")
            return False
    except Exception as e:
        print(f"❌ LLM client error: {e}")
        return False

def run_workflow(prompt, file_path):
    """
    Runs the full document retrieval and data extraction workflow.
    """
    try:
        print("\n" + "="*60)
        print("🚀 STARTING WORKFLOW")
        print("="*60)

        # 1. Document Retrieval
        print("\n[Step 1/3] Retrieving documents...")
        doc_retriever = DocRetrieval(file_path, prompt)
        doc_retriever.doc_retrieval()

        if not doc_retriever.info_map:
            print("❌ No articles were found or processed. Stopping workflow.")
            return None
        
        print(f"✓ Retrieved {len(doc_retriever.info_map)} articles.")

        # 2. Context Extraction
        print("\n[Step 2/3] Extracting data from documents...")
        extractor = Extract()
        
        extracted_df = extractor.context_extract(doc_retriever)

        if extracted_df is None or extracted_df.empty:
            print("❌ Initial data extraction resulted in an empty dataset.")
            return None

        print("✓ Initial data extraction complete.")
        
        # 3. Data Quality Verification
        print("\n[Step 3/3] Verifying and refining data quality...")
        verifier = Verify()
        max_retries = 3
        is_quality_sufficient = False
        final_df = extracted_df
        sample_df = doc_retriever.sample_df

        for i in range(max_retries):
            print(f"\n--- Verification Attempt {i + 1}/{max_retries} ---")
            
            # The verify_data_quality function will regenerate if needed
            current_df, is_quality_sufficient = verifier.verify_data_quality(
                extracted_df=final_df,
                sample_df=sample_df,
                context_extractor=extractor,
                doc_retrieval_instance=doc_retriever
            )
            final_df = current_df # Update to the latest dataframe

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

def main():
    """Main function to run the document retrieval test."""
    print("🚀 Starting Clinical Trial Document Retrieval Test")
    
    load_environment()
    
    if not validate_environment():
        return
    
    if not test_llm_connection():
        return
    
    prompt, file_path = get_user_input()
    if not prompt or not file_path:
        return

    final_dataframe = run_workflow(prompt, file_path)

    if final_dataframe is not None:
        print("\n" + "="*60)
        print("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nFinal Extracted Data:")
        pd.set_option('display.max_rows', 50)
        pd.set_option('display.max_columns', 20)
        pd.set_option('display.width', 120)
        print(final_dataframe)
        
        save = input("\nSave the resulting DataFrame to a CSV file? (y/n): ").strip().lower()
        if save == 'y':
            output_filename = "extracted_data.csv"
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
        print("\n\n👋 Test interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
