from llm import ask_gpt_oss

def generate_answer(query, context):
    print("[Tool] Calling GPT-OSS...")

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question ONLY from the provided context.

Question:
{query}

Context:
{context}

Answer:
"""

    response = ask_gpt_oss(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response
