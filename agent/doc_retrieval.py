from typing import Any
import pandas as pd
import json
from .llm_util import LLMClient
import os
from Bio import Entrez
import xml.etree.ElementTree as ET
from .verification import Verify

class DocRetrieval:
    def __init__(self, file: str, prompt: str):
        self.sample_df = pd.read_csv(file)
        self.user_prompt = prompt
        self.num_articles = 0
        self.info_map = {} 


    #from the prompt and the csv file create a search query prompt
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

    #define the prompt with the file and the user prompt
    def create_prompt(self):
        prompt = f"""You are an expert biomedical researcher and librarian, specializing in crafting advanced search queries for PubMed. Your task is to convert a user's PICO-based request into a highly effective, structured PubMed search query.

        User Request (PICO format): "{self.user_prompt}"

        **Instructions:**
        1.  **Deconstruct the PICO:** First, break down the user's PICO request into its core semantic concepts.
        2.  **Identify MeSH Terms:** For each core concept, identify the most relevant Medical Subject Headings (MeSH terms). This is critical for a high-quality search.
        3.  **Combine with Field Tags:** Create search groups for each concept by combining the MeSH term (`[mh]`) with relevant keywords in the Title/Abstract (`[tiab]`) using the `OR` operator.
        4.  **Use Boolean Logic:** Combine the concept groups using the `AND` operator. Use parentheses `()` to ensure the logic is correctly nested.
        5.  **Balance Specificity and Breadth:** The query should be specific enough to find relevant articles but broad enough to ensure results. Using `OR` between MeSH and `[tiab]` terms is the best way to achieve this.
        6.  **Final Output:** Return ONLY the final, single-line search query string. Do not include explanations or any other text.

        **Example:**
        - **PICO Request:** {{'P': 'Patients with type 2 diabetes', 'I': 'Metformin', 'C': 'Placebo', 'O': 'Effect on blood sugar levels'}}
        - **Generated Query:** (("Diabetes Mellitus, Type 2"[mh] OR "type 2 diabetes"[tiab]) AND ("Metformin"[mh] OR "metformin"[tiab])) AND ("Blood Glucose"[mh] OR "glycemic control"[tiab] OR "blood sugar"[tiab])

        Now, generate the PubMed search query for the provided User Request.
        """
        
        return prompt

    def search_pmc(self, term, retmax=15):
        """
        Search PubMed Central for articles matching a term, sorted by relevance.
        Returns a list of PMC IDs.
        """
        try:
            # The default sort order when the 'sort' parameter is not specified is relevance ('Best Match').
            handle = Entrez.esearch(db="pmc", term=term, retmax=retmax, retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            return records["IdList"]
        except Exception as e:
            print(f"Error searching PMC: {e}")
            return []


    def fetch_summary(self, pmc_ids):
        """
        Get metadata (title, authors, full text) for PMC articles in batches.
        """
        if not pmc_ids:
            return

        batch_size = 20  # Process 20 IDs at a time to avoid URL length issues
        for i in range(0, len(pmc_ids), batch_size):
            batch_ids = pmc_ids[i:i+batch_size]
            ids_str = ",".join(batch_ids)
            
            try:
                # 1. Fetch summaries (metadata)
                handle_summary = Entrez.esummary(db="pmc", id=ids_str, retmode="xml")
                summary_records = Entrez.read(handle_summary)
                handle_summary.close()

                # 2. Fetch full text XML
                handle_fetch = Entrez.efetch(db="pmc", id=ids_str, rettype="full", retmode="xml")
                xml_text = handle_fetch.read()
                handle_fetch.close()
                
                # 3. Parse full text and map to PMCID
                full_texts = {}
                root = ET.fromstring(xml_text)
                for article_node in root.findall('.//article'):
                    article_id_node = article_node.find(".//article-id[@pub-id-type='pmc']")
                    if article_id_node is not None:
                        article_pmcid = article_id_node.text
                        body = article_node.find('body')
                        article_text = ""
                        if body is not None:
                            paragraphs = body.findall('.//p')
                            # Join non-empty paragraph texts
                            article_text = "\n".join([p.text for p in paragraphs if p.text and p.text.strip()])
                        full_texts[article_pmcid] = article_text

                # 4. Combine metadata and full text
                for record in summary_records:
                    self.num_articles += 1
                    article_key = self.num_articles
                    pmc_id = record['Id']
                    
                    self.info_map[article_key] = {
                        "pmc_id": pmc_id,
                        "title": record.get('Title', 'No title found'),
                        "authors": record.get('AuthorList', []),
                        "text": full_texts.get(pmc_id, "")  # Get the mapped full text
                    }
            except Exception as e:
                print(f"Warning: Batch fetch failed for IDs {ids_str}. Error: {e}")
        return
    


    #use the prompt to search for docs through the pubmed api and retrieve them
    def doc_retrieval(self):
        Entrez.email = os.getenv('PUBMED_EMAIL')
        Entrez.api_key = os.getenv('ENTREZ_API_KEY')

        #call create_prompt method
        prompt = self.create_prompt()
        query = self.llm_extract(prompt)
        old_queries = [query]
        
        max_retries = 10
        pmc_ids = []

        
        for attempt in range(max_retries):
            print(f"Search attempt {attempt + 1}/{max_retries} with query: \"{query}\"")
            pmc_ids = self.search_pmc(query)

            if pmc_ids:
                print(f"Found {len(pmc_ids)} PMC articles.")
                break

            if attempt < max_retries - 1:
                print("No articles found. Attempting to generate a new query.")
                verifier = Verify()
                query = verifier.recreate_query(self.user_prompt, self.sample_df, old_queries)
                old_queries.append(query)
        
        if not pmc_ids:
            print("No articles found after multiple attempts. Returning empty result.")
            return {}
        
        print('about to call fetch summary')
        self.fetch_summary(pmc_ids)
        
        return self.info_map
