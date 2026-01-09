import json
import pandas as pd
from typing import TYPE_CHECKING
from .llm_util import LLMClient

if TYPE_CHECKING:
    from .context_extract import Extract
    from .doc_retrieval import DocRetrieval

class Verify:
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

        messages = chat_prompt

        completion = llm_client.client.chat.completions.create(
            model=llm_client.deployment,
            messages=messages,
            max_tokens=1000,
            temperature=0.2, # Slightly more creative for a new query
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )

        return json.loads(completion.to_json())["choices"][0]["message"]["content"]

    def _clean_json_from_llm(self, response_text: str) -> str:
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

    def recreate_query(self, user_prompt: str, sample_df: pd.DataFrame, previous_queries: list[str]):
        """
        Creates a new, broader PubMed query if the previous one returned no results by strategically simplifying the PICO structure.
        """
        
        prompt = f"""You are an expert PubMed query troubleshooter. The following structured PubMed query returned ZERO results, likely because it was too specific. Your task is to intelligently broaden the query to get some results.

        The best way to broaden a query is to remove or simplify the least critical PICO concepts. Follow this hierarchy of simplification:
        1.  **Attempt 1 (Remove Outcome):** First, try removing the 'Outcome' part of the query. The 'Population' and 'Intervention' are most important.
        2.  **Attempt 2 (Remove Comparison):** If that still fails, remove the 'Comparison' part as well.
        3.  **Attempt 3 (Simplify Intervention/Population):** As a last resort, simplify the 'Intervention' or 'Population' clauses by removing a restrictive keyword or using a broader term.

        **Original User Request (PICO):** "{user_prompt}"
        **Previous Failed Queries:** "{previous_queries}"

        ---
        **Example:**
        - **Failed Query:** (("Diabetes Mellitus, Type 2"[mh] OR "type 2 diabetes"[tiab]) AND ("Metformin"[mh] OR "metformin"[tiab])) AND ("Blood Glucose"[mh] OR "glycemic control"[tiab])
        - **Analysis:** This query includes Population, Intervention, and Outcome. The first step is to remove the Outcome.
        - **New, Broader Query:** (("Diabetes Mellitus, Type 2"[mh] OR "type 2 diabetes"[tiab]) AND ("Metformin"[mh] OR "metformin"[tiab]))
        ---

        Based on the last failed query, generate the next logical attempt at a broader query. Do not repeat a previous query. Return ONLY the new query string.
        """

        return self.llm_extract(prompt)

    def analyze_data_quality(self, extracted_df: pd.DataFrame, sample_df: pd.DataFrame) -> dict:
        """
        Compares the extracted dataset to the sample dataset to analyze relevancy and structure.
        Returns a dictionary with a score and analysis.
        """
        if extracted_df.empty:
            return {
                "score": 0.0,
                "analysis": "The extracted dataset is empty."
            }

        prompt = f"""You are a data quality assurance expert. Your task is to analyze an extracted dataset by comparing it to a sample dataset.
        You need to evaluate the structural integrity and data relevancy of the extracted data.

        Provide a score from 0.0 to 1.0, where 1.0 represents a perfect match in structure and high data relevancy.
        Also provide a brief analysis explaining your score.

        Sample Dataset (for structure and type reference):
        {sample_df.to_json(orient='records', lines=True)}

        Extracted Dataset (to be evaluated):
        {extracted_df.to_json(orient='records', lines=True)}

        ---
        RULES FOR EVALUATION:
        1.  **Structural Similarity (Weight: 35%)**:
            *   Do the column names in the extracted data match the sample data?
            *   Are the data types appropriate for the columns as suggested by the sample data? (e.g., numeric columns should contain numbers).
            *   Is the overall structure a valid list of JSON objects?

        2.  **Data Relevancy & Quality (Weight: 65%)**:
            *   How much of the extracted data is null or empty? A high proportion of nulls is negative.
            *   Does the extracted data seem plausible and relevant based on the kind of data in the sample? (e.g., if sample has ages 20-50, an extracted age of 500 is wrong).
            *   Is the content of the extracted data consistent with what you would expect for those columns?

        ---
        OUTPUT FORMAT:
        Return a single JSON object with two keys: "score" (a float between 0.0 and 1.0) and "analysis" (a string).

        Example Output:
        {{
            "score": 0.85,
            "analysis": "The extracted data has a matching structure and the data types are correct. However, the 'Dosage (mg)' column contains a few null values, slightly reducing the score. The data is highly relevant."
        }}

        ---
        Now, perform the analysis.
        """
        
        response_json_str = self.llm_extract(prompt)
        try:
            cleaned_json = self._clean_json_from_llm(response_json_str)
            return json.loads(cleaned_json)
        except json.JSONDecodeError:
            return {
                "score": 0.0,
                "analysis": f"Failed to parse LLM response as JSON: {response_json_str}"
            }

    def regenerate_data(self, context_extractor: "Extract", doc_retrieval_instance: "DocRetrieval") -> pd.DataFrame:
        """
        Triggers the data extraction process again to regenerate the dataset.
        """
        print("Regenerating data for more relevant extraction and accurate structure...")
        return context_extractor.context_extract(doc_retrieval_instance)

    def verify_data_quality(self, extracted_df: pd.DataFrame, sample_df: pd.DataFrame, context_extractor: "Extract", doc_retrieval_instance: "DocRetrieval", quality_threshold: float = 0.75):
        """
        Verifies the quality of the extracted data. If the quality is below a threshold,
        it triggers a regeneration of the data.
        
        Returns the final DataFrame and a boolean indicating if the quality was sufficient.
        """
        analysis_results = self.analyze_data_quality(extracted_df, sample_df)
        quality_score = analysis_results.get("score", 0.0)
        analysis = analysis_results.get("analysis", "No analysis provided.")

        print(f"Data Quality Score: {quality_score:.2f}/1.0")
        print(f"Analysis: {analysis}")

        is_sufficient = quality_score >= quality_threshold
        final_df = extracted_df

        if not is_sufficient:
            final_df = self.regenerate_data(context_extractor, doc_retrieval_instance)
            print("Data has been regenerated.")
        else:
            print("Data quality meets the threshold.")
            
        return final_df, is_sufficient
