import pandas as pd
import json
import torch
from typing import Dict, List, Any
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class LEADSExtract:
    MODEL_NAME = "zifeng-ai/leads-mistral-7b-v1"

    def __init__(self, info_map: Dict[int, Dict[str, Any]], semantic_cols: List[str]):
        """
        Initialize the LEADS semantic extractor.

        Args:
            info_map: Dictionary mapping article keys to article data
                      (pmc_id, title, authors, text) — populated by DocRetrieval.
            semantic_cols: List of semantic column names to extract from each article.
                           Defined by the calling script.
        """
        self.info_map = info_map
        self.semantic_cols = semantic_cols
        self.output_df = pd.DataFrame(columns=self.semantic_cols)

        # Load model and tokenizer once on init.
        # from_pretrained() automatically caches to ~/.cache/huggingface/hub/
        # so the download only happens on the first run.
        print(f"Loading LEADS model ({self.MODEL_NAME})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

        self.device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"  Using device: {self.device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            dtype=torch.float16 if self.device != "cpu" else torch.float32,
        ).to(self.device)
        self.model.eval()
        print("✓ LEADS model loaded.")

    def _build_prompt(self, paper_text: str) -> str:
        """
        Build an instruction prompt for the LEADS model to extract
        the target semantic columns from the given paper text.
        """
        cols_str = ", ".join(self.semantic_cols)

        prompt = (
            f"You are a clinical data extraction expert. "
            f"Extract the following fields from the research paper text below: {cols_str}.\n\n"
            f"For each field, extract the most relevant value directly stated in the text. "
            f"If a field is not found, use \"N/A\".\n\n"
            f"Research Paper Text:\n{paper_text}\n\n"
            f"Return your answer as a single JSON object with the field names as keys."
        )
        return prompt

    def _generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Run inference on the LEADS model using the Mistral instruct format.
        Truncates input to fit within the model's 2048-token context window.
        """
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt"
        ).to(self.model.device)

        # Truncate input to fit within model's context window (2048 tokens)
        max_input_len = 2048 - max_new_tokens
        if input_ids.shape[-1] > max_input_len:
            print(f"  Truncating input from {input_ids.shape[-1]} to {max_input_len} tokens")
            input_ids = input_ids[:, :max_input_len]

        # Create attention mask (1 for all real tokens)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the newly generated tokens (skip the input)
        generated_tokens = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the model's response text into a dictionary.
        Attempts to find and parse a JSON object from the output.
        """
        # Try to find JSON object boundaries
        start = response_text.find('{')
        end = response_text.rfind('}')

        if start != -1 and end != -1 and end > start:
            json_str = response_text[start:end + 1]
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    return {col: parsed.get(col, "N/A") for col in self.semantic_cols}
            except json.JSONDecodeError:
                pass

        # If JSON parsing fails, return N/A for all columns
        print(f"Warning: Could not parse LEADS model output as JSON: {response_text[:200]}...")
        return {col: "N/A" for col in self.semantic_cols}

    def extract_semantic_data(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract semantic data from a single article using the LEADS model.

        Args:
            article: Dictionary containing article data, must have a 'text' key.

        Returns:
            A dictionary mapping each semantic column to its extracted value.
        """
        paper_text = article.get("text", "")

        if not paper_text or not paper_text.strip():
            return {col: "N/A" for col in self.semantic_cols}

        prompt = self._build_prompt(paper_text)
        response = self._generate(prompt)
        return self._parse_response(response)

    def process_article(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Wraps extract_semantic_data with error handling.

        Returns:
            A list containing a single extracted row dict, or an empty list on failure.
        """
        try:
            row = self.extract_semantic_data(article)
            return [row]
        except Exception as e:
            print(f"Warning: LEADS extraction failed for article "
                  f"'{article.get('title', 'Unknown')}': {e}")
            return []

    def run_extraction(self) -> pd.DataFrame:
        """
        Run LEADS semantic extraction on all articles in info_map sequentially.
        (Sequential because the model runs on GPU and concurrent inference
        on the same model instance would cause issues.)

        Returns:
            A DataFrame with one row per article and columns matching semantic_cols.
        """
        all_rows: List[Dict[str, Any]] = []
        articles = list(self.info_map.values())

        for article in tqdm(articles, desc="LEADS semantic extraction"):
            rows = self.process_article(article)
            if rows:
                all_rows.extend(rows)

        if all_rows:
            self.output_df = pd.DataFrame(all_rows).reindex(
                columns=self.semantic_cols, fill_value=None
            )
        else:
            self.output_df = pd.DataFrame(columns=self.semantic_cols)

        return self.output_df
