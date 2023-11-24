from dotenv import load_dotenv, find_dotenv
import os
import requests
import pandas as pd

_ = load_dotenv(find_dotenv())


class HFEmbeddings:
    def __init__(self):
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        hf_token = os.getenv("HUGGINGFACE_API_KEY")

        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
        self.headers = {"Authorization": f"Bearer {hf_token}"}

    def query(self, texts):
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": texts, "options": {"wait_for_model": True}})
        return response.json()


texts = ["How do I get a replacement Medicare card?",
         "What is the monthly premium for Medicare Part B?"]

if __name__ == "__main__":
    obj = HFEmbeddings()
    output = obj.query(texts)
    embeddings = pd.DataFrame(output)
    print(embeddings)
