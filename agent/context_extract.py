import json
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from .llm_util import LLMClient
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

if TYPE_CHECKING:
    from .doc_retrieval import DocRetrieval

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
        self.col_descriptions = {}  # Auto-generated column descriptions from sample data
        self.output_df = pd.DataFrame()
        self.user_prompt = ""
        self.consistency_pairs = []

    def extract_data(self, context, sample_df):
        user_prompt = self.user_prompt
        prompt = f"""You are a clinical data extraction expert. Your task is to extract structured data from the provided research paper text.

        here's the user prompt that describes the type and kind of clinical data that is supposed to be extracted to help you make better decisions: {user_prompt}

        You must extract data for the following columns:
        {self.cols}

        Use the sample data below to understand the exact required output format, data types, and stylistic conventions for each column. This is your primary reference for how to structure and phrase every output.

        Sample Data:
        {sample_df.to_json(orient='records')}

        Research Paper Text:
        {context}

        ---
        RULES:

        ### RULE 1 — STRICT FORMAT AND STYLE MIRRORING (HIGHEST PRIORITY)
        For every column, carefully study the sample data to determine:
        - **Structure:** Is the value a plain string/number, or a dict with `"value"` and `"context"` keys? Mirror this exactly.
        - **Style and length:** Is the value brief and precise (e.g., just "ELISA"), or detailed and descriptive? Mirror the style and level of detail shown in the sample for that specific column.
        - **Data type:** Is it a percentage, integer, free text? Match exactly.

        When a column uses dict format, the fields must be:
        - `"value"`: the extracted value (number, string, or "N/A").
        - `"context"`: a **direct verbatim quote** from the Research Paper Text that supports the extracted value. If the value is N/A, set context to "N/A".

        ### RULE 2 — STRONG N/A BIAS (CRITICAL FOR ACCURACY)
        You must apply a strong conservative bias toward outputting N/A. Specifically:
        - **Only extract a value if it is explicitly and unambiguously stated in the Research Paper Text.**
        - If you are uncertain, if the information must be inferred, or if the text only vaguely implies a value, output N/A.
        - **Never hallucinate, guess, or construct a value that is not directly supported by the text.**
        - A confident N/A is always better than a uncertain extraction. Maximize true positives and minimize false positives — a false positive is worse than a missed extraction.
        - For columns where the sample frequently shows N/A (e.g., `method`), be especially conservative: only extract if the specific method is explicitly named in the text.

        ### RULE 3 — N/A REPRESENTATION
        Represent missing values as:
        - `{{"value": "N/A", "context": "N/A"}}` for dict-format columns.
        - `"N/A"` for plain-format columns.
        - Never omit a key. Never leave a field blank.

        ### RULE 4 — NUMERICAL EXTRACTION
        - If a percentage is explicitly stated (e.g., "37%"), use it directly.
        - If only a ratio is given (e.g., "58/156"), calculate: round to 2 decimal places.
        - Output only the number for numeric columns (e.g., `1500`, not `"1500mg"`).

        ### RULE 5 — SEMANTIC AND TEXT FIELDS
        - Extract the most complete and accurate description directly supported by the text.
        - Mirror the length and style of the sample for that column — do not over-explain if the sample shows brief values, and do not truncate if the sample shows detailed values.
        - The `"context"` field must always be a direct verbatim quote from the text, never a paraphrase.

        ### RULE 6 — OUTPUT FORMAT
        Output the extracted data as a JSON list of objects, one object per row of data found. Each object must contain all target columns.

        ---
        Now perform the extraction on the provided Research Paper Text.
        """
        return self.llm_extract(prompt)

    def generate_column_descriptions(self, sample_df):
        prompt = f"""You are a clinical data expert. Given a sample dataset and extraction goal, do two things:

        Goal: {self.user_prompt}
        Columns: {self.cols}
        Sample data: {sample_df.head(3).to_json(orient='records')}

        1. For each column, write a 1-sentence description of what it represents.

        2. Identify consistency pairs — columns that are so closely related that 
        if one is found the other should almost always also be found. 
        Two types:
        - "rate_count": a percentage/rate column paired with its raw count column 
        (e.g., percentage positive paired with number positive)
        - "anchor_detail": a study identifier column paired with any detail column 
        about that same study (e.g., trial_name paired with n_study)
        Only include pairs you are confident about from the sample data.

        Return a JSON object with two keys:
        - "descriptions": dict of column name to 1-sentence description
        - "consistency_pairs": list of objects like 
        {{"type": "rate_count", "cols": ["percentage", "n_positive"]}}

        Return ONLY the JSON, no other text.
        """
        raw = self.llm_extract(prompt)
        try:
            result = json.loads(clean_json_from_llm(raw))
            if isinstance(result, list) and result:
                result = result[0]
            if isinstance(result, dict):
                self.col_descriptions = result.get("descriptions", {})
                self.consistency_pairs = result.get("consistency_pairs", [])
            else:
                raise ValueError("Parsed JSON is not a dictionary")
        except Exception as e:
            print(f"Warning: Could not parse column descriptions properly: {e}")
            self.col_descriptions = {col: col for col in self.cols}
            self.consistency_pairs = []

    def extract_core_text(self, context, sample_df):
        user_prompt = self.user_prompt

        # Build column descriptions block
        if self.col_descriptions:
            col_info = "\n".join([f"- **{col}**: {desc}" for col, desc in self.col_descriptions.items()])
        else:
            col_info = "\n".join([f"- {col}" for col in self.cols])

        prompt = f"""You are a clinical data extraction expert. Your task is to read a full research paper and write a concise prose summary containing all the data points needed for structured extraction.

        Extraction goal: {user_prompt}

        Target columns to extract (what data each column needs):
        {col_info}

        Sample data showing what a good context summary looks like and the expected values for each column:
        {sample_df.to_json(orient='records')}

        Research Paper Text:
        {context}

        ---
        RULES:

        ### RULE 1 — UNIFIED NARRATIVE (HIGHEST PRIORITY)
        Write a single prose summary paragraph covering all data relevant to the extraction goal and target columns. Do NOT organize by column names, do NOT use headers like "## percentage" or "## n_study". Write it as a natural, flowing paragraph — similar in style to the context fields shown in the sample data.

        ### RULE 2 — COLUMN-DRIVEN COMPLETENESS
        For each target column, find the relevant data point in the paper and include it in your summary. Study the column descriptions and sample data to understand exactly what each column needs:
        - If a column expects a number, find and include that number.
        - If a column expects a name or identifier, find and include it.
        - If a column expects a description, include the relevant description.
        Every target column should be addressed in your summary. If the paper truly has no data for a column, that is fine — do not fabricate data.

        ### RULE 3 — VERBATIM NUMBERS
        All numbers, percentages, p-values, dosages, and statistical values must be copied exactly from the paper. Do not round, paraphrase, or calculate. If the paper says "37.5%", write "37.5%", not "about 38%".

        ### RULE 4 — SAME-STUDY ONLY
        Only include data from THIS study's own results. Do not include results cited from other published studies, background literature, or comparison references. If the paper says "Previous studies showed 15% incidence", do NOT include that — only include this paper's own findings.

        ### RULE 5 — NEGATIVE FINDINGS ARE DATA
        Explicit statements of absence are valid and important data points. Phrases like "no responses were detected", "0% incidence", "none of the patients developed antibodies" MUST be included in the summary. Do NOT skip them.

        ### RULE 6 — SYNONYM AWARENESS
        Clinical papers may use different terminology to describe the same concept. Look for synonyms, abbreviations, and alternative phrasings that map to your target columns. For example:
        - "anti-drug antibodies" = "ATAs" = "anti-therapeutic antibodies" = "immunogenicity"
        - "overall survival" = "OS", "adverse events" = "AEs" = "toxicities"
        - "enrolled" = "randomized" = "included in the analysis"
        Use the column descriptions and sample data to understand which terms map to your columns.

        ### RULE 7 — SAMPLE-GUIDED STYLE AND LENGTH
        Study the sample data to understand the expected format. Your summary should be similar in length and detail to the context fields shown in the sample data. If sample contexts are detailed multi-sentence paragraphs, write a detailed paragraph. If they are brief, write briefly.

        ---
        Now write the prose context summary for the provided Research Paper Text.
        """
        return self.llm_extract(prompt)

    def verify_context(self, extracted_context, sample_df):
        """Verify and clean the extracted prose context. Checks accuracy, relevance, and internal consistency."""
        user_prompt = self.user_prompt

        # Build column descriptions block
        if self.col_descriptions:
            col_info = "\n".join([f"- **{col}**: {desc}" for col, desc in self.col_descriptions.items()])
        else:
            col_info = "\n".join([f"- {col}" for col in self.cols])

        prompt = f"""You are a clinical data verification expert. Your task is to review a prose context summary extracted from a research paper and ensure it is accurate, relevant, and internally consistent.

        Extraction goal: {user_prompt}

        Target columns (what data will be extracted from this context):
        {col_info}

        Sample data showing expected values for each column:
        {sample_df.to_json(orient='records')}

        Extracted context to verify:
        {extracted_context}

        Consistency pairs (columns that should logically co-occur):
        {self.consistency_pairs}

        ---
        RULES:

        ### RULE 1 — PRESERVE FORMAT
        Output the verified context as the same prose narrative format. Do NOT reorganize into columns, headers, or bullet points. Keep it as a flowing paragraph.

        ### RULE 2 — KEEP BY DEFAULT
        Your default action is to KEEP every sentence. Only remove or modify content that is clearly wrong, fabricated, or irrelevant to the extraction goal. Do not remove content just because you are unsure — when in doubt, keep it.

        ### RULE 3 — CONSISTENCY CHECK
        Use the consistency pairs to verify internal logic:
        - If a rate/percentage is stated, its corresponding count column should be mathematically plausible.
        - If a study name is given, the enrollment numbers should correspond to that same study.
        - If one column in a pair has data but its partner is completely absent from the context, flag this as a concern — but do NOT fabricate the missing data. Just ensure the present data is accurate.

        ### RULE 4 — WRONG-STUDY REMOVAL
        Remove any data that clearly comes from a different study cited within the paper (background references, prior literature), not from the study being analyzed. Clues: "Previous studies showed...", "In a prior trial...", "Smith et al. reported...".

        ### RULE 5 — NEGATIVE FINDINGS ARE VALID
        Do NOT remove explicit negative results. Statements like "no responses were detected", "0% incidence", "none of the patients developed antibodies" are valid and important data points. They must be preserved.

        ---
        Now output the verified prose context summary.
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
        user_prompt = self.user_prompt
        prompt = f"""  You are an expert data normalization assistant. Your task is to process a sample dataset and
        ensure it contains the three essential columns: paper title, author, and source evidence.

        here's the user prompt that describes the type and kind of clinical data that is supposed to be extracted to help you make better decisions: {user_prompt}

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
        user_prompt = self.user_prompt
        prompt = f"""You are an expert data transformation assistant. Your task is to create a new dataset based on a sample dataset and a new list of columns.

        here's the user prompt that describes the type and kind of clinical data that is supposed to be extracted to help you make better decisions: {user_prompt}

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

    def process_article(self, article, df):
        context = article.get('text', 'no text found')
        core_text = self.extract_core_text(context, df)
        verified_text = self.verify_context(core_text, df)
        new_data_json_raw = self.extract_data(verified_text, df)
        
        try:
            cleaned_json = clean_json_from_llm(new_data_json_raw)
            new_data = json.loads(cleaned_json)
            
            if isinstance(new_data, dict):
                return [new_data]
            elif isinstance(new_data, list):
                return new_data
        except (json.JSONDecodeError, TypeError):
            print(f"Warning: Could not parse or process LLM output for an article: {new_data_json_raw}")
        
        return []

    def context_extract(self, doc_retrieval_instance: "DocRetrieval"):
        df = doc_retrieval_instance.sample_df
        self.user_prompt = doc_retrieval_instance.user_prompt
        if not self.cols:
            try:
                cols_list_raw = self.column_extract(df)
                self.cols = json.loads(clean_json_from_llm(cols_list_raw))
                # Initialize output_df with the determined columns, but as an empty DataFrame
                self.output_df = pd.DataFrame(columns=self.cols)
            except Exception as e:
                print(f"Exception: {e}")

        # Generate column descriptions once (if not already done)
        if not self.col_descriptions and self.cols:
            print("Generating column descriptions from sample data...")
            self.generate_column_descriptions(df)

        all_new_rows = []
        articles = list(doc_retrieval_instance.info_map.values())
        
        with ThreadPoolExecutor() as executor:
            # Use a lambda to pass the dataframe `df` to the worker function
            future_to_article = {executor.submit(self.process_article, article, df): article for article in articles}
            
            for future in tqdm(as_completed(future_to_article), total=len(articles), desc="Extracting data"):
                try:
                    new_rows = future.result()
                    if new_rows:
                        all_new_rows.extend(new_rows)
                except Exception as exc:
                    article = future_to_article[future]
                    print(f"An article generated an exception: {article.get('title', 'Unknown Title')}: {exc}")

        if all_new_rows:
            new_rows_df = pd.DataFrame(all_new_rows)
            # Assign directly instead of concatenating with potentially existing sample data
            self.output_df = new_rows_df.reindex(columns=self.cols, fill_value=None)
        else:
            # If no new rows are extracted, ensure output_df is still an empty DataFrame with correct columns
            self.output_df = pd.DataFrame(columns=self.cols)
        
        return self.output_df
            

 


