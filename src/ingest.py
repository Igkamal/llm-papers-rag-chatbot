from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Paths (work no matter where you run the script from)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DATA"
DB_DIR = BASE_DIR / "chroma_db"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    # Step 1: Load all PDFs
    print("Loading PDFs...")
    docs = PyPDFDirectoryLoader(str(DATA_DIR)).load()
    print(f"Loaded {len(docs)} pages")

    # Step 2: Split into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    # Step 3: Create the embedding model
    print("Loading embedding model (first time downloads ~90 MB)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Step 4: Embed the chunks and save them in ChromaDB
    print("Building vector store (this may take a few minutes)...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )
    print(f"Done! Vector store saved in: {DB_DIR}")


if __name__ == "__main__":
    main()