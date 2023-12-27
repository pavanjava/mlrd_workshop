from llama_index.llms import Ollama, ChatMessage

llm = Ollama(base_url="http://localhost:11434", model="codellama")

while True:
    query = input("\nEnter your prompt: \n")
    if query == 'quit':
        break
    else:
        messages = [
            ChatMessage(
                role="system",
                content="you are a python code expert, your job is to assist users with coding related questions\n"
            ),
            ChatMessage(
                role="user", content=query
            )
        ]
    resp_stream = llm.stream_chat(messages=messages)

    for r in resp_stream:
        print(r.delta, end="")
