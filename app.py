from pathlib import Path
import streamlit as st
from src.rag_chain import ask

st.set_page_config(page_title="LLM Papers RAG Chatbot", page_icon="📚")
st.title("📚 LLM Research Papers Chatbot")
st.caption("Ask questions about Transformers, BERT, GPT-3, RAG, LoRA, LLaMA and more. "
           "Answers come only from the 8 papers in the knowledge base.")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("RAG pipeline: PDFs → chunks → embeddings → ChromaDB → "
             "retrieval → Groq LLM → answer")
    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()

# Keep chat history between interactions
if "history" not in st.session_state:
    st.session_state.history = []

# Show previous messages
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['file']}**, page {s['page']}")
                    st.caption(s["text"])

# New question
if question := st.chat_input("Ask a question about the papers..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the papers..."):
            answer, docs = ask(question)
        st.write(answer)

        sources = [
            {
                "file": Path(d.metadata.get("source", "")).name,
                "page": d.metadata.get("page", 0) + 1,   # PDF pages start at 0 in metadata
                "text": d.page_content[:300] + "...",
            }
            for d in docs
        ]
        with st.expander("Sources"):
            for s in sources:
                st.markdown(f"**{s['file']}**, page {s['page']}")
                st.caption(s["text"])

    st.session_state.history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )