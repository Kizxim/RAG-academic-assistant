from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

DB_DIR = "./chroma_db"
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
OLLAMA_MODEL = "llama3.2:3b"


def load_vector_db():
    if not Path(DB_DIR).exists():
        raise FileNotFoundError("Chroma DB not found. Run ingest.py first.")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=DB_DIR,
    )


def format_sources(docs):
    seen = set()
    sources = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown source")
        start_index = doc.metadata.get("start_index", "")

        key = (source, start_index)
        if key in seen:
            continue

        seen.add(key)
        sources.append(f"- {source} | start_index: {start_index}")

    return "\n".join(sources)


def answer_question(query, vector_db, llm):
    results = vector_db.similarity_search(query, k=4)

    context = "\n\n".join(
        [f"Source: {doc.metadata.get('source')}\n{doc.page_content}" for doc in results]
    )

    prompt = f"""
You are an academic study assistant.

Use only the context below to answer the question.
If the context does not contain enough information, say that the provided documents do not contain enough information.
Answer clearly and concisely.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, results


def main():
    vector_db = load_vector_db()

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.2,
    )

    print("AcademicAssistant")
    print("Type your question, 'exit' to quit.")
    print()

    while True:
        query = input("Ask: ").strip()

        if query.lower() in ["exit", "quit"]:
            break

        if not query:
            continue

        answer, sources = answer_question(query, vector_db, llm)

        print("\nAnswer:")
        print(answer)

        print("\nSources:")
        print(format_sources(sources))
        print()


if __name__ == "__main__":
    main()