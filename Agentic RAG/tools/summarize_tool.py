from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2:7b"
)

def summarize_text(text):

    prompt = f"""
Summarize the following context clearly.

Context:
{text}
"""

    response = llm.invoke(prompt)

    return response.content