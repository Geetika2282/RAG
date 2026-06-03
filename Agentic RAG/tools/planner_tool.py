from llm import ask_gpt_oss


def decide_action(query, observation):

    prompt = f"""
You are an autonomous RAG agent.

Question:
{query}

Observation:
{observation}

Available tools:

RETRIEVE
SUMMARIZE
ANSWER

Think carefully.

Respond with:

Thought: ...
Action: RETRIEVE

OR

Thought: ...
Action: SUMMARIZE

OR

Thought: ...
Action: ANSWER
"""

    response = ask_gpt_oss([
        {
            "role": "user",
            "content": prompt
        }
    ])

    print("\n===== PLANNER OUTPUT =====")
    print(response)
    print("==========================\n")

    if response is None:
        return "Action: RETRIEVE"

    return response.strip()