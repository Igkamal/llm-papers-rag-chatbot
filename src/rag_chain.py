from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"   # if this errors, use "llama-3.1-8b-instant"

# Load the saved vector store
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectordb = Chroma(
    persist_directory=str(BASE_DIR / "chroma_db"),
    embedding_function=embeddings,
)
retriever = vectordb.as_retriever(search_kwargs={"k": 6})

# The LLM
llm = ChatGroq(model=LLM_MODEL, temperature=0)

# The prompt: forces the model to use only the retrieved context
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant that answers questions about Large Language Model research papers.
Answer ONLY using the context below.
If the answer is not in the context, say exactly: "I don't have enough information in the documents to answer that."

Context:
{context}

Question: {question}

Answer:"""
)


def ask(question: str):
    """Retrieve relevant chunks, then generate a grounded answer."""
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    answer = (prompt | llm).invoke({"context": context, "question": question}).content
    return answer, docs


if __name__ == "__main__":
    q = "What is the self-attention mechanism?"
    answer, docs = ask(q)
    print("\nQUESTION:", q)
    print("\nANSWER:", answer)
    print("\nSOURCES:")
    for d in docs:
        print("-", Path(d.metadata.get("source", "")).name, "page", d.metadata.get("page"))