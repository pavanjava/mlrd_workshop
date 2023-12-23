from llama_index.llms import Ollama, ChatMessage

llm = Ollama(model="mistral", base_url="http://localhost:11434")

messages = [
    ChatMessage(
        role="system", content="you are a multi lingual assistant used for translation and your job is to translate nothing more than that."
    ),
    ChatMessage(
        role="user", content="please translate message in triple tick to french ``` What is standard deviation?```"
    )
]
resp_stream = llm.stream_chat(messages=messages)

for r in resp_stream:
    print(r.delta, end="")
