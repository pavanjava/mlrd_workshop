from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import MilvusVectorStore
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


class MilvusVectorizer:
    def __init__(self):
        collection_name: str = 't55_product_specs'
        db_uri: str = 'http://localhost:19530'
        # load documents
        documents = SimpleDirectoryReader(input_dir='/Users/pavanmantha/Pavans/Workshops/Mallareddy-university/mlrd_workshop/tutorials/section-4/data', required_exts=['.pdf']).load_data()
        vector_store = MilvusVectorStore(dim=1536, overwrite=False, collection_name=collection_name, uri=db_uri) # overwrite=True will override existing vectors in the store.
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        service_context = ServiceContext.from_defaults(chunk_size=1536, chunk_overlap=200)
        # save the documents index to vector store
        self.index = VectorStoreIndex.from_documents(service_context=service_context, storage_context=storage_context, documents=documents)

    def query(self, user_query):
        query_engine = self.index.as_query_engine()
        response = query_engine.query(str_or_query_bundle=user_query)
        print(response.response)


if __name__ == "__main__":
    obj = MilvusVectorizer()
    obj.query("What is the length, Height of the all engines ?")



