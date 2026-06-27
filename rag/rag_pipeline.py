from vectorstores.retriever import retrieve_chunks
from rag.prompts import RAG_PROMPT
from rag.llm_client import call_llm

def run_rag(query: str, top_k: int = 5):
    chunks = retrieve_chunks(query, top_k)

    if not chunks:
        return "No relevant information found."

    context = "\n\n".join(chunk["text"] for chunk in chunks)

    prompt = RAG_PROMPT.format(
        context=context,
        question=query
    )

    return call_llm(prompt)
