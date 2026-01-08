import json
import pandas as pd
from .llm_util import LLMClient

def _clean_json_from_llm(response_text: str) -> str:
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

class AgentEvaluator:
    def llm_call(self, prompt: str):
        """Helper function to make a call to the LLM."""
        llm_client = LLMClient()
        chat_prompt = [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
        completion = llm_client.client.chat.completions.create(
            model=llm_client.deployment,
            messages=chat_prompt,
            max_tokens=2000,
            temperature=0.1,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        return json.loads(completion.to_json())["choices"][0]["message"]["content"]

    def evaluate_extraction(self, extracted_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> dict:
        """
        Compares the extracted dataset to the ground truth dataset to score performance.
        Returns a dictionary with scores and analysis.
        """
        if extracted_df.empty:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "relevancy_score": 0.0,
                "analysis": "The extracted dataset is empty, but the ground truth was not."
            }

        prompt = f"""You are a data quality assurance expert. Your task is to evaluate an extracted dataset by comparing it to a ground truth dataset.
        You need to calculate precision, recall, and F1-score for the structured data, and provide a relevancy score for the overall content.

        - **Precision**: What percentage of extracted items are correct (present in the ground truth)?
        - **Recall**: What percentage of ground truth items were successfully extracted?
        - **Relevancy Score**: A score from 0.0 to 1.0 on how relevant the extracted data is to the ground truth, considering context and meaning, not just exact matches.

        Ground Truth Dataset (The correct data):
        {ground_truth_df.to_json(orient='records', lines=True)}

        Extracted Dataset (The data to be evaluated):
        {extracted_df.to_json(orient='records', lines=True)}

        ---
        RULES FOR EVALUATION:
        1.  **Matching**: Consider rows a match if the key data points are equivalent, even if formatting differs slightly.
        2.  **Calculations**:
            - True Positives (TP): Number of rows in extracted data that are also in the ground truth.
            - False Positives (FP): Number of rows in extracted data that are NOT in the ground truth.
            - False Negatives (FN): Number of rows in ground truth that are NOT in the extracted data.
            - Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            - Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            - F1-Score = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0
        3.  **Relevancy**: Assess if the extracted data captures the same meaning and information as the ground truth, even if values aren't identical (e.g., synonyms, different units that convert to the same value).

        ---
        OUTPUT FORMAT:
        Return a single JSON object with keys: "precision", "recall", "f1_score", "relevancy_score", and "analysis".

        Example Output:
        {{
            "precision": 0.8,
            "recall": 0.67,
            "f1_score": 0.73,
            "relevancy_score": 0.85,
            "analysis": "The extracted data correctly identified 4 out of 5 ground truth items. One extracted item was incorrect. Recall is lower because two ground truth items were missed. The data is highly relevant."
        }}

        ---
        Now, perform the evaluation.
        """
        
        response_json_str = self.llm_call(prompt)
        try:
            cleaned_json = _clean_json_from_llm(response_json_str)
            return json.loads(cleaned_json)
        except json.JSONDecodeError:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "relevancy_score": 0.0,
                "analysis": f"Failed to parse LLM response as JSON: {response_json_str}"
            }
