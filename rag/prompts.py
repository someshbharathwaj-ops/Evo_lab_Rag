# RAG/prompts.py

RAG_PROMPT = """
You are a helpful assistant.

Answer the question using ONLY the information provided in the context below.
If the answer is not present in the context, say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
