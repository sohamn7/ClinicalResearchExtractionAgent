import json
import pandas as pd
from llm_util import LLMClient

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

    def recreate_query(self, user_prompt: str, sample_df: pd.DataFrame, previous_queries: list[str]):
        """
        Creates a new PubMed query if the previous one returned no results.
        """
        
        prompt = f"""You are an expert search query creator. The previous PubMed search query failed to return any results. Your task is to generate a new, revised query.

        Previous failed queries: "{previous_queries}"

        Analyze the original user request and sample data to create a broader or alternative query. Try using different keywords, simplifying the query, or using broader MeSH terms.

        Original User Request: "{user_prompt}"
        Sample Data Columns: {sample_df.columns.tolist()}

        ---
        Instructions:
        1.  Generate a new, revised PubMed search query.
        2.  The query should be a concise string of keywords and operators.
        3.  Return ONLY the new search query string.
        ---

        Generate the new PubMed search query.
        """

        return self.llm_extract(prompt)