import json
from typing import Dict, List, Any, Optional
from llm_util import LLMClient
from doc_retrieval import DocRetrieval
import pandas as pd
from tqdm import tqdm

def clean_json_from_llm(response_text: str) -> str:
    """
    Extracts a JSON string from a markdown code block if present.
    It finds the first '[' or '{' and the last ']' or '}' to extract the JSON content.
    """
    # Find the start of the JSON, whether it's a list or an object
    start_index = response_text.find('[')
    if start_index == -1:
        start_index = response_text.find('{')

    # Find the end of the JSON
    end_index = response_text.rfind(']')
    if end_index == -1:
        end_index = response_text.rfind('}')

    # If both a start and an end are found, slice the string
    if start_index != -1 and end_index != -1:
        return response_text[start_index : end_index + 1]
    
    # If no valid JSON structure is found, return the original string
    return response_text

class Extract:
    def __init__(self):
        self.cols = []
        self.output_df = pd.DataFrame()
        

    def extract_data(self, context, sample_df):
        prompt = f"""You are a data extraction expert. Your task is to extract structured data from the provided research paper text.

        You must extract data for the following columns with included units for extraction. Pay close attention to the units specified
        in the column names and convert the extracted data to match:
        {self.cols}

        Use the sample data below to understand the desired format and data types for each column.

        Sample Data:
        {sample_df.to_json(orient='records')}

        Research Paper Text:
        {context}

        ---
        RULES:
        1. Extract all data points that match the columns.
        2. Convert units to match the column names (e.g., if a column is 'Dosage (mg)' and the text says '2g', you must convert it to '2000').
        3. If a value for a column is not found, use `null`.
        4. Output the extracted data as a JSON object. If multiple data points are found, return a list of JSON objects.

        ---
        Example 1:
        - Columns: ["Drug", "Dosage (mg)"]
        - Text: "The patient was administered 1.5g of Paracetamol."
        - Output: [{{'Drug': 'Paracetamol', 'Dosage (mg)': 1500}}]

        Example 2:
        - Columns: ["Patient Age (years)", "Temperature (Celsius)"]
        - Text: "A 45-year-old male presented with a fever of 101°F."
        - Output: [{{'Patient Age (years)': 45, 'Temperature (Celsius)': 38.3}}]

        ---

        Now, perform the extraction on the provided Research Paper Text.
        """
        return self.llm_extract(prompt)

    def llm_extract(self, prompt):
        llm_client = LLMClient()
        
        chat_prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        # Include speech result if speech is enabled
        messages = chat_prompt

        # Generate the completion
        completion = llm_client.client.chat.completions.create(
            model= llm_client.deployment,
            messages=messages,
            max_tokens=10000,
            temperature=0.05,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )

        return json.loads(completion.to_json())["choices"][0]["message"]["content"]

    def column_extract(self, df):
        
        prompt = f"""  You are an expert data normalization assistant. Your task is to process a sample dataset and
        ensure it contains the three essential columns: paper title, author, and source evidence.

        You will be given a JSON list of column names. Follow these rules:

        1. Identify and Rename Synonyms:
            * If there is a column name that is a synonym for paper title (e.g., 'article_name', 'title', 'document'), rename
                it to paper title.
            * If there is a column name that is a synonym for author (e.g., 'creator', 'writer', 'researcher'), rename it to
                author.
            * If there is a column name that is a synonym for source evidence (e.g., 'citation', 'reference',
                'source_document'), rename it to source evidence.

        2. Add Missing Columns:
            * If, after checking for synonyms, paper title is not in the list, add it.
            * If, after checking for synonyms, author is not in the list, add it.
            * If, after checking for synonyms, source evidence is not in the list, add it.

        3. Preserve Other Columns: Keep all other original column names in the list unchanged.

        4. Output: Return only the final, modified list of column names as a single JSON array of strings.

        Example 1:
        * Input: ["article_name", "publication_year", "creator"]
        * Output: ["paper title", "publication_year", "author", "source evidence"]

        Example 2:
        * Input: ["title", "author", "journal", "citation"]
        * Output: ["paper title", "author", "journal", "source evidence"]

        Example 3:
        * Input: ["PatientID", "Drug", "Outcome"]
        * Output: ["PatientID", "Drug", "Outcome", "paper title", "author", "source evidence"]

        ---

        Your Input Sample Dataset:
        {df}  """

        return self.llm_extract(prompt)
    
    def copy_dataset(self, cols_list, df):
        prompt = f"""You are an expert data transformation assistant. Your task is to create a new dataset based on a sample dataset and a new list of columns.

        You will be given a sample dataset and a JSON list of target column names.

        Follow these rules:
        1. Create a new dataset that is a copy of the original sample dataset.
        2. The new dataset must have the columns specified in the target list of column names.
        3. For columns that exist in the original dataset, copy the data.
        4. For columns that are in the target list but not in the original dataset, add them and leave their values empty.
        5. Return the new dataset as a JSON object.

        ---

        Sample Dataset:
        {df.to_json(orient='records')}

        Target Columns:
        {cols_list}
        """

        return self.llm_extract(prompt)

    def context_extract(self, doc_retrieval_instance: DocRetrieval):
        df = doc_retrieval_instance.sample_df
        if self.cols == []:
            try:
                cols_list_raw = self.column_extract(df)
                # print(f"llm output columns: {cols_list_raw}")
                self.cols = json.loads(clean_json_from_llm(cols_list_raw))
                # print(f"columns: {self.cols}")

                copy_dataset_raw = self.copy_dataset(cols_list_raw, df)
                self.output_df = pd.DataFrame(json.loads(clean_json_from_llm(copy_dataset_raw)))
            except Exception as e:
                print(f"Exception: {e}")

        all_new_rows = []
        # Assuming info_map values are the article dictionaries
        for article in tqdm(doc_retrieval_instance.info_map.values(), desc="Extracting data"):
            context = article.get('text', 'no text found')
            
            new_data_json_raw = self.extract_data(context, df)
            
            try:
                # The LLM can return a single dict or a list of dicts
                cleaned_json = clean_json_from_llm(new_data_json_raw)
                new_data = json.loads(cleaned_json)
                
                if isinstance(new_data, dict):
                    all_new_rows.append(new_data)
                elif isinstance(new_data, list):
                    all_new_rows.extend(new_data)
            except (json.JSONDecodeError, TypeError):
                # Handle cases where LLM output is not valid JSON or not a dict/list
                print(f"Warning: Could not parse or process LLM output for an article: {new_data_json_raw}")

        if all_new_rows:
            new_rows_df = pd.DataFrame(all_new_rows)
            self.output_df = pd.concat([self.output_df, new_rows_df], ignore_index=True)
        
        return self.output_df
            

 


