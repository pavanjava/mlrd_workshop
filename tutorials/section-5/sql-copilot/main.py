from fastapi import FastAPI
from payload import Body
from text2sql import GenerateSQL

description = """
Natural Language to SQL API helps you to generate complex SQL from just natural language. 🚀

## Payload Description
- query: natural language query asked by user
- model: LLM model under the hood (possible values: Mistral, codellama, llama2)
- include_tables (optional): list of table names
"""

app = FastAPI(title="Natural Language to SQL",
              description=description,
              version="1.0.0",
              contact={
                  "Author": "Pavan Kumar Mantha",
              },
              license_info={
                  "name": "Apache 2.0",
                  "identifier": "MIT",
              }, )
sql_generator = GenerateSQL()


@app.post("/generate-query")
def generate_query(request: Body):
    return sql_generator.generate(query=request.query, model=request.model, include_tables=request.include_tables)
