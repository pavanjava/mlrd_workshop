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
        if helper.fetch_collection("t5-product-spec") is not None:
            self.chroma_collection = helper.fetch_collection("t5-product-spec")
        else:
            self.chroma_collection = helper.create_collection("t5-product-spec")

        # define embedding function
        embed_model = OpenAIEmbedding(api_key=os.getenv('OPENAI_API_KEY'))

        # load documents
        self.documents = SimpleDirectoryReader(
            input_dir='/section-4/data',
            required_exts=['.pdf']).load_data()
        # set up ChromaVectorStore and load in data
        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # initialise service context with default values
        self.service_context = ServiceContext.from_defaults(embed_model=embed_model, chunk_size=1000, chunk_overlap=200)

    def save_to_database(self):
        index = VectorStoreIndex.from_documents(
            self.documents, storage_context=self.storage_context, service_context=self.service_context)
        return index

    def query(self, index):
        # Query Data
        query_engine = index.as_query_engine()
        response = query_engine.query("The T55 fleet has accumulated how many hours of operation?")
        pprint(response.response)


if __name__ == "__main__":
    object = ChromaDBVectorizer()
    index = object.save_to_database()
    object.query(index=index)
