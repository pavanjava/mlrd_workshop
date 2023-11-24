import faiss
from llama_index import SimpleDirectoryReader, load_index_from_storage, VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from dotenv import load_dotenv, find_dotenv
from pprint import pprint

_ = load_dotenv(find_dotenv())


class FaissVectorizer:
    def __init__(self):
        # dimensions of text-ada-embedding-002
        dim = 1536
        faiss_index = faiss.IndexFlatL2(dim)
        # load documents
        documents = SimpleDirectoryReader(input_dir='/Users/pavanmantha/Pavans/PracticeExamples/DataScience_Practice/LLMs/llama_index_tutorials/vector-stores/data',required_exts=['.pdf']).load_data()
        # create storage_context and faiss_index
        self.vector_store = FaissVectorStore(faiss_index=faiss_index)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_documents(documents, storage_context=self.storage_context)
        # save index to disk
        self.index.storage_context.persist()

    def load_from_index(self):
        # load index from disk
        self.vector_store = FaissVectorStore.from_persist_dir("./storage")
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store, persist_dir="./storage")
        self.index = load_index_from_storage(storage_context=self.storage_context)
        return self.index

    def query(self):
        # set Logging to DEBUG for more detailed outputs
        query_engine = self.index.as_query_engine()
        response = query_engine.query("The T55 fleet has accumulated how many hours of operation?")
        pprint(response.response)


if __name__ == '__main__':
    obj = FaissVectorizer()
    obj.query()
