from openai import AzureOpenAI
import os

class LLMClient:
    def __init__(self):
        try:
            self.azure_endpoint = os.getenv("ENDPOINT_URL")
            self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
            self.deployment = os.getenv("DEPLOYMENT_NAME")
            # self.api_version = api_version

            if not all([self.azure_endpoint, self.api_key, self.deployment]):
                raise RuntimeError("Azure OpenAI credentials or endpoint not configured. Check your .env file.")

            # Configure openai for Azure
            self.client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version="2024-05-01-preview",
            )

        except Exception as e:
            # Log the error with details but continue execution
            error_type = type(e).__name__
            error_msg = str(e)
            
            print(f"API Error ({error_type}): {error_msg}...") 
            
            # For context length exceeded errors, log how much we exceeded by
            if "context length" in error_msg.lower():
                print("Context length exceeded - try reducing the prompt size")
                
            # Re-raise the exception to be handled by the caller
            raise e

