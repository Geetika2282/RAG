from tools.chunk_tool import chunk_text
from tools.embed_tool import create_vector_store
from tools.retrieve_tool import retrieve_context
from tools.summarize_tool import summarize_text
from tools.answer_tool import generate_answer
from tools.planner_tool import decide_action

import re


class RagAgent:

    def process_document(self, text):

        print("\n[Agent] Chunking document...")

        chunks = chunk_text(text)

        print(f"[Agent] Created {len(chunks)} chunks")

        print("[Agent] Creating vector store...")

        create_vector_store(chunks)

        print("[Agent] Document processed successfully.")

   

    def ask(self, query):

        observation = "No context retrieved yet."

        for step in range(5):

            action = decide_action(
                query,
                observation
            )
            print(f"\n[Planner Decision]")
            print(action)

            print(f"\n[Agent Thought] {action}")

            if "RETRIEVE" in action:

                context = retrieve_context(query)

                observation = "\n".join(context)

                print("\n[Retrieved]")
                print(observation[:500])

            elif "SUMMARIZE" in action:

                observation = summarize_text(
                    observation
                )

            elif "ANSWER" in action:

                return generate_answer(
                    query,
                    observation
                )

        return "Agent exceeded max steps."
