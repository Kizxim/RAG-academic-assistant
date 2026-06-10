import re
import shutil
import warnings
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

warnings.filterwarnings("ignore", category=DeprecationWarning)

RESOURCE_DIR = "./resource"
DB_DIR = "./chroma_db"
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


def clean_text(text: str) -> str:
    text = text.replace("\t", " ")

    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if re.fullmatch(r"[xX*=+\-_/\\|().\s]+", line):
            continue

        if re.fullmatch(r"\d{1,3}", line):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def load_documents():
    loader = DirectoryLoader(
        path=RESOURCE_DIR,
        glob="**/*.*",
        loader_cls=UnstructuredFileLoader,
        show_progress=True,
        silent_errors=True,
    )

    docs = loader.load()

    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    docs = [doc for doc in docs if len(doc.page_content.strip()) > 50]

    print(f"Loaded documents: {len(docs)}")
    return docs


def split_documents(docs):
    markdown_separators = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=650,
        chunk_overlap=150,
        add_start_index=True,
        strip_whitespace=True,
        separators=markdown_separators,
        is_separator_regex=True,
    )

    chunks = text_splitter.split_documents(docs)
    print(f"Created chunks: {len(chunks)}")
    return chunks


def build_vector_db(chunks):
    db_path = Path(DB_DIR)

    if db_path.exists():
        shutil.rmtree(db_path)
        print("Removed old Chroma DB")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIR,
    )

    print("Created new Chroma DB")


def main():
    docs = load_documents()
    chunks = split_documents(docs)
    build_vector_db(chunks)


if __name__ == "__main__":
    main()