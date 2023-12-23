import chromadb
from chromadb.api.models.Collection import Collection
from pprint import pprint

class ChromaDBHelper:
    def __init__(self):
        self.client = chromadb.HttpClient(host="localhost", port="8000")

    def fetch_all_collections(self):
        return self.client.list_collections()

    def fetch_collection(self, collection_name: str):
        return self.client.get_collection(name=collection_name)

    def create_collection(self, collection_name: str):
        return self.client.create_collection(name=collection_name)

    def delete_collection(self, collection_name: str):
        self.client.delete_collection(name=collection_name)

    def fetch_embeddings(self, collection: Collection):
        return collection.get(limit=5)



if __name__ == "__main__":
    helper = ChromaDBHelper()

    collections = helper.fetch_all_collections()
    for collection in collections:
        print(collection)
        embeddings = helper.fetch_embeddings(collection=collection)
        pprint(embeddings)



