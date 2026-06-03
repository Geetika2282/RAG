from llm import ask_gpt_oss

def summarize_text(text):
    print("[Tool] Summarizing...")

    prompt = f"""
Summarize the following context while preserving key information.

{text}
"""

    response = ask_gpt_oss(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response
