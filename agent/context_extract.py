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
        """Generate 1-sentence descriptions per column from the sample data. Called once per run."""
        prompt = f"""You are a clinical data expert. Given a sample dataset and a description of the extraction goal, describe what each column represents.

        Goal: {self.user_prompt}

        Columns: {self.cols}

        Sample data (first few rows):
        {sample_df.head(3).to_json(orient='records')}

        ---
        For each column, write a 1-sentence description of:
        - What type of value it holds (number, percentage, name, free text, etc.)
        - What it specifically represents in the context of the extraction goal

        Return your answer as a JSON object where keys are column names and values are the 1-sentence descriptions.
        Example: {{"percentage": "The incidence rate (%) of the target outcome in the study population", "n_study": "Total number of patients enrolled or randomized in the primary study"}}

        Return ONLY the JSON object, no other text.
        """
        raw = self.llm_extract(prompt)
        try:
            self.col_descriptions = json.loads(clean_json_from_llm(raw))
        except (json.JSONDecodeError, TypeError):
            print(f"Warning: Could not parse column descriptions. Using column names as fallback.")
            self.col_descriptions = {col: col for col in self.cols}

    def extract_core_text(self, context, sample_df):
        user_prompt = self.user_prompt

        # Build column descriptions block
        if self.col_descriptions:
            col_info = "\n".join([f"- **{col}**: {desc}" for col, desc in self.col_descriptions.items()])
        else:
            col_info = "\n".join([f"- {col}" for col in self.cols])

        prompt = f"""You are a clinical immunogenicity data analyst. Your task is to find verbatim quotes from a research paper to populate structured extraction columns.

        ===== EXTRACTION GOAL =====
        {user_prompt}

        ===== TARGET COLUMNS =====
        {col_info}

        ===== SAMPLE DATA (shows expected format and values for each column) =====
        {sample_df.to_json(orient='records')}

        ===== RESEARCH PAPER TEXT =====
        {context}

        ===================================================================
        COLUMN CATEGORIES — different extraction rules apply:

        GROUP A — ADA IMMUNOGENICITY COLUMNS (must cite explicit ADA data):
        - percentage: The ADA positivity rate (%) in this study's treatment population
        - n_positive: Number of patients who developed anti-drug antibodies in this study
        - method: The specific named assay used to detect ADA (e.g., ELISA, ECL assay)

        GROUP B — STUDY METADATA COLUMNS (general study info, no ADA keyword required):
        - n_study: Total number of patients enrolled/randomized/treated in this study
        - trial_name: The formal name, acronym, or ClinicalTrials.gov ID of this study
        - patient_characteristics: Actual reported demographics (age, sex, disease status)
        - dosage: The specific drug dose and schedule administered to patients
        - notes: Any ADA-specific observation (titers, transient positivity, clinical impact of ADA)

        ===================================================================
        ABSOLUTE EXCLUSION LIST — NEVER extract these for GROUP A columns (percentage, n_positive):
        - Objective response rates (ORR), partial/complete responses (PR, CR), disease control rates
        - Adverse event (AE) rates, cytokine release syndrome (CRS), infusion reactions, toxicity grades
        - Progression-free survival (PFS), overall survival (OS)
        - Circulating tumor cell (CTC) counts or biomarker positivity unrelated to ADA
        - Pharmacokinetic (PK) data (Cmax, AUC, half-life, drug concentrations)
        - Data from OTHER studies cited within this paper (background references, comparison studies)

        If the quote does not EXPLICITLY mention ADA, immunogenicity, or anti-[drug name] antibodies for GROUP A — write NOT FOUND.

        ===================================================================
        TWO-STEP PROCESS:

        ### STEP 1 — IDENTIFY SECTIONS
        For GROUP A columns (percentage, n_positive, method):
        - Look for sections explicitly about immunogenicity, ADA, or antibody detection
        - If no such section exists, write NONE

        For GROUP B columns (n_study, trial_name, patient_characteristics, dosage, notes):
        - Look in Methods, Study Design, Patients, Results, and Introduction sections
        - IMPORTANT: Nearly every clinical paper states an enrollment number — do NOT write NONE for n_study unless the paper truly has none
        - IMPORTANT: Always look for a study name/acronym/NCT number for trial_name

        Write: **column_name** → [section or NONE] — [brief reason]

        ### STEP 2 — EXTRACT VERBATIM QUOTES
        Using ONLY the sections in Step 1:

        For GROUP A (percentage, n_positive, method):
        - Quote MUST explicitly mention ADA/immunogenicity AND cite a specific number/assay
        - Negative ADA results are valid: "no ADA detected", "all patients remained ADA-negative"
        - Write NOT FOUND if no ADA-specific quote exists

        For GROUP B (n_study, trial_name, patient_characteristics, dosage, notes):
        - Extract the clearest verbatim sentence
        - n_study: Use the ITT/FAS/safety-analysis-set count (typically stated near start of Results)
        - trial_name: Look for study acronyms, "the [Name] study", or NCT numbers in Introduction or Methods
        - patient_characteristics: Must be ACTUAL reported demographics (e.g., "median age was 58 years, 71% male"), NOT eligibility criteria ("patients aged ≥18")
        - dosage: Specific amount and schedule (e.g., "150 mg SC every 4 weeks")
        - notes: An ADA-specific observation if present; otherwise NOT FOUND
        - Write NOT FOUND only if the paper truly has no content for the column

        ===================================================================
        OUTPUT FORMAT:

        ### STEP 1: Section Relevance
        **column_name** → [section or NONE] — [reason]

        ### STEP 2: Extracted Quotes

        ## [Column Name]
        <verbatim quote, or NOT FOUND>

        (one section per column, same order as target columns)
        ===================================================================
        Now extract from the Research Paper Text above.
        """
        return self.llm_extract(prompt)

    def verify_context(self, extracted_context, sample_df):
        """Two-tier verification: strict for ADA columns, permissive for metadata columns."""
        user_prompt = self.user_prompt

        # Build column descriptions block
        if self.col_descriptions:
            col_info = "\n".join([f"- **{col}**: {desc}" for col, desc in self.col_descriptions.items()])
        else:
            col_info = "\n".join([f"- {col}" for col in self.cols])

        prompt = f"""You are a clinical data verification expert reviewing extracted quotes from a research paper.

        Extraction goal: {user_prompt}

        Column definitions:
        {col_info}

        Sample data showing expected values:
        {sample_df.to_json(orient='records')}

        ===== EXTRACTED QUOTES TO VERIFY =====
        {extracted_context}

        ===================================================================
        COLUMN GROUPS — apply different verification standards:

        GROUP A — ADA IMMUNOGENICITY COLUMNS (strict verification):
        - percentage, n_positive, method

        GROUP B — STUDY METADATA COLUMNS (lenient verification):
        - n_study, trial_name, patient_characteristics, dosage, notes

        ===================================================================
        VERIFICATION RULES:

        FOR GROUP A COLUMNS (percentage, n_positive, method) — STRICT:
        REJECT (replace with NOT FOUND) if the quote:
        - Does NOT explicitly mention ADA, immunogenicity, anti-drug antibodies, or anti-[drug name] antibodies
        - Is about response rates, adverse events, survival, PK data, or CTCs instead of ADA
        - Is from another study cited within the paper (not this paper's own results)
        - For 'method': does NOT name a SPECIFIC assay (generic phrases like "immunogenicity was assessed" are REJECTED)
        - For 'percentage'/'n_positive': does NOT state a specific number or percentage for ADA positivity
        KEEP if the quote explicitly and directly reports ADA/immunogenicity data for this study.

        FOR GROUP B COLUMNS (n_study, trial_name, patient_characteristics, dosage, notes) — LENIENT:
        REJECT only if the quote is CLEARLY WRONG:
        - n_study: REJECT only if the number is clearly a subgroup/screening count, not total enrollment
        - trial_name: REJECT only if the quote has no trial name, acronym, or NCT number at all
        - patient_characteristics: REJECT only if the quote is eligibility criteria ("patients aged ≥18 were eligible") rather than actual reported demographics
        - dosage: REJECT only if no specific dose amount is mentioned
        - notes: REJECT only if the quote is a completely non-ADA general statement totally unrelated to immunogenicity
        WHEN IN DOUBT FOR GROUP B — KEEP the quote.

        ===================================================================
        OUTPUT FORMAT:
        Output ONLY the ## [Column Name] sections.
        Each section contains either the original quote (if kept) or NOT FOUND (if rejected).
        Do NOT include STEP 1 analysis, commentary, or explanations.
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
            

 


