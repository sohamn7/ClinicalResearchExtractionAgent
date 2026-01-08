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
        prompt = f"""You are an expert at creating search queries for the PubMed Entrez search engine. Your task is to generate a precise and effective search query based on a user's request and a sample dataset.

        User Request: "{self.user_prompt}"
        Sample Data Columns: {self.sample_df.columns.tolist()}

        Instructions:
        1.  Identify the main keywords and concepts from the User Request and the Sample Data Columns.
        4.  The query should be a concise string of keywords and operators. Avoid natural language questions.
        5.  Return ONLY the search query string.

        Examples of BAD queries:
        - "What are new treatments for diabetes?"
        - "diabetes treatment"

        Now, generate the PubMed search query. Make sure not to make the prompt too specific, so PMC articles can be matched.
        """
        
        return prompt

    def search_pmc(self, term):
        """
        Search PubMed Central for articles matching a term.
        Returns a list of PMC IDs.
        """
        try:
            handle = Entrez.esearch(db="pmc", term=term, retmax=50, retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            return records["IdList"]
        except Exception as e:
            print(f"Error searching PMC: {e}")
            return []


    def fetch_summary(self, pmc_ids):
        """
        Get metadata (title, authors, journal) for PMC articles.
        Returns a list of records (dicts).
        """
        if not pmc_ids:
            return []
        
        ids = ",".join(pmc_ids)
        handle = Entrez.esummary(db="pmc", id=ids, retmode="xml")
        records = Entrez.read(handle)
        # print(f'printed records: {records}')
        handle.close()

        handle = Entrez.efetch(db="pmc", id=ids, rettype="full", retmode="xml")
        xml_text = handle.read()
        # print(f"print xml_text: {xml_text}")
        handle.close()

        root = ET.fromstring(xml_text)

        for record in records:
            self.num_articles += 1
            article_key = self.num_articles
            
            title = record.get('Title', 'No title found')
            authors = record.get('AuthorList', [])
            
            # Find the corresponding article in the XML by PMC ID
            pmc_id = record['Id']
            article_text = ""
            for article in root.findall('.//article'):
                body = article.find('body')
                if body is not None:
                    paragraphs = body.findall('.//p')
                    article_text = "\n".join([p.text for p in paragraphs if p.text])

            self.info_map[article_key] = {
                "pmc_id": pmc_id,
                "title": title,
                "authors": authors,
                "text": article_text
            }

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
            return json.dumps({"papers": {}, "total_papers": 0, "error": "No articles found"})
        
        print('about to call fetch summary')
        self.fetch_summary(pmc_ids)
        
        return self.info_map
