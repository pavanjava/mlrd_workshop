import chromadb


class ChromaDBHelper:
    def __init__(self):
        self.client = chromadb.HttpClient(host="localhost", port="8000")

    def fetch_collection(self, collection_name):
        return self.client.get_collection(name=collection_name)

    def create_collection(self, collection_name):
        return self.client.create_collection(name=collection_name)

    def delete_collection(self, collection_name):
        self.client.delete_collection(name=collection_name)


if __name__ == "__main__":
    helper = ChromaDBHelper()
    collection = helper.fetch_collection()
    if helper.fetch_collection() is not None:
        print(collection.name)
