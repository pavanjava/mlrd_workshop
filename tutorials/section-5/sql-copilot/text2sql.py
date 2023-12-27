import os
import logging as logger
from typing import List
from dotenv import load_dotenv, find_dotenv
from sqlalchemy import create_engine
from llama_index.indices.struct_store import NLSQLTableQueryEngine
from llama_index.embeddings import OllamaEmbedding, LangchainEmbedding
from llama_index.llms import Ollama
from llama_index import SQLDatabase, ServiceContext

_ = load_dotenv(find_dotenv())
logger.basicConfig(level=logger.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pg_uri = f"postgresql+psycopg2://{os.getenv('DB_USER_NAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{int(os.getenv('DB_PORT'))}/{os.getenv('DATABASE_NAME')}"
logger.info(pg_uri)


class GenerateSQL():
    def __init__(self):
        self.engine = create_engine(url=pg_uri)

    def generate(self, query: str, model: str, include_tables: List[str] | None = None):
        llm = Ollama(base_url=os.getenv('LLM_URI'), model=model)
        # embed_model = OllamaEmbedding(base_url=os.getenv('LLM_URI'), model_name=model)
        embed_model = LangchainEmbedding(langchain_embeddings='LCEmbeddings', model_name=model)
        service_context = ServiceContext.from_defaults(llm=llm,
                                                       embed_model=embed_model)  # "local:BAAI/bge-small-en-v1.5"
        sql_database = SQLDatabase(self.engine, include_tables=include_tables)

        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            service_context=service_context,
        )
        logger.info(f'Query: {query}')
        resp = query_engine.query(query)
        logger.info(f'Response: {resp.response}')
        logger.info(f'Query: {resp.metadata["sql_query"]}')
        return {'response': resp.response, 'metadata': {'query': resp.metadata['sql_query']}}


# standalone method to test the functionality
if __name__ == '__main__':
    sql_generator = GenerateSQL()
    response = sql_generator.generate(query='which all shipper shipped maximum orders', model='mistral')
    print(response)
