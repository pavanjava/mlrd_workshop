from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, OpenAIEmbedding
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from dotenv import load_dotenv, find_dotenv
from chromadb_helper import ChromaDBHelper
from pprint import pprint
import os

_ = load_dotenv(find_dotenv())


class ChromaDBVectorizer:
    def __init__(self):
        helper = ChromaDBHelper()
        collection_name: str = "<your_collection_name>"
        try:
            if helper.fetch_collection(collection_name) is not None:
                self.chroma_collection = helper.fetch_collection(collection_name=collection_name)
            else:
                self.chroma_collection = helper.create_collection(collection_name=collection_name)
        except Exception as e:
            self.chroma_collection = helper.create_collection(collection_name=collection_name)
            print("Error")
        # define embedding function
        embed_model = OpenAIEmbedding(api_key=os.getenv('OPENAI_API_KEY'))

        # load documents
        self.documents = SimpleDirectoryReader(
            input_dir='<path_to_knowledge_base_dir>',
            required_exts=['.pdf']).load_data()
        # set up ChromaVectorStore and load in data
        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # initialise service context with default values
        self.service_context = ServiceContext.from_defaults(embed_model=embed_model,
                                                            chunk_size=1000,
                                                            chunk_overlap=200)

    def save_to_database(self):
        index = VectorStoreIndex.from_documents(
            self.documents, storage_context=self.storage_context, service_context=self.service_context)
        return index

    def query(self, index):
        # Query Data
        query_engine = index.as_query_engine()
        response = query_engine.query("what is the segment profit of aerospace?")
        pprint(response.response)


if __name__ == "__main__":
    object = ChromaDBVectorizer()
    index = object.save_to_database()
    object.query(index=index)
