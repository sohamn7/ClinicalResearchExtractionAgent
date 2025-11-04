# Clinical Trial Document Retrieval and Extraction System

This system retrieves relevant clinical trial documents from PubMed, extracts comprehensive metadata and full text, and processes the information using LLM for structured data extraction based on user prompts and sample data.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file with the following variables:

```env
# Azure OpenAI Configuration
ENDPOINT_URL=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
DEPLOYMENT_NAME=your-deployment-name

# PubMed/Entrez Configuration
ENTREZ_API_KEY=your-entrez-api-key
PUBMED_EMAIL=your-email@example.com
```

### 3. Get API Keys

**Azure OpenAI:**
- Create an Azure OpenAI resource
- Get your endpoint URL and API key
- Create a deployment (e.g., GPT-4)

**PubMed/Entrez:**
- Get a free API key from NCBI: https://www.ncbi.nlm.nih.gov/account/
- Use your email address for the PUBMED_EMAIL variable

## Usage

Run the main script:
```bash
python main.py
```

The script will:
1. Validate your environment variables
2. Test the LLM connection
3. Prompt you for:
   - Your clinical trial data requirements (prompt)
   - Path to your CSV file
4. Process the data and retrieve relevant documents from PubMed
5. Extract comprehensive metadata and full text from each paper
6. Create a structured JSON map with paper information
7. Use LLM to extract relevant information based on your requirements
8. Save the extracted results to a JSON file

## Sample Data

A sample CSV file (`sample_data.csv`) is included for testing. It contains patient data with columns:
- PatientID,Drug,Condition,Outcome

## Features

### Document Retrieval (`doc_retrieval.py`)
- Searches PubMed Central for relevant articles
- Extracts comprehensive metadata including:
  - Title, authors, journal, publication date
  - Abstract, keywords, DOI, PMC ID, PMID
  - Full text content (methods, results, conclusions)
  - References
- Creates structured JSON map for LLM processing
- Handles rate limiting and error management

### Context Extraction (`context_extract.py`)
- Processes JSON paper data using LLM
- Extracts relevant information based on user requirements
- Supports custom extraction prompts
- Saves results in structured JSON format
- Handles specific data type extraction

## Example Prompts

- "Find studies on diabetes treatment with metformin"
- "Research hypertension management in elderly patients"
- "Clinical trials for insulin therapy effectiveness"
- "Studies on cardiovascular outcomes in diabetic patients"

## Output Structure

The system generates two main outputs:

1. **Paper Map JSON**: Contains all retrieved papers with metadata
2. **Extraction Results JSON**: Contains LLM-processed relevant information

Both files are saved with timestamps for easy tracking.
