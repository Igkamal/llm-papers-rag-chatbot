import sys, types, time
from pathlib import Path

# RAGAS 0.4.3 imports a module that newer langchain-community removed.
# This tiny stub lets RAGAS load. Do not delete it.
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI: pass
_stub.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _stub

from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import ResponseRelevancy
from ragas.run_config import RunConfig

from rag_chain import ask, llm, embeddings

BASE_DIR = Path(__file__).resolve().parent.parent

# ---- 10 questions with reference answers (CHECK THESE AGAINST YOUR PDFs) ----
QA = [
    {"q": "What does the Transformer use instead of recurrence and convolution?",
     "ref": "It relies entirely on attention mechanisms (self-attention), which allows much more parallelization."},
    {"q": "What is masked language modeling in BERT?",
     "ref": "Some input tokens are randomly masked and the model is trained to predict the masked tokens using context from both directions."},
    {"q": "What is few-shot learning in GPT-3?",
     "ref": "The model is given a few task demonstrations in its context at inference time, without any gradient updates or fine-tuning."},
    {"q": "What are the two components of a RAG model?",
     "ref": "A retriever that fetches relevant documents and a sequence-to-sequence generator that produces the answer conditioned on them."},
    {"q": "What are the two variants of RAG?",
     "ref": "RAG-Sequence, which uses the same retrieved document for the whole output, and RAG-Token, which can use different documents per token."},
    {"q": "What is chain-of-thought prompting?",
     "ref": "Prompting a language model with examples that include intermediate reasoning steps, so it produces step-by-step reasoning before the final answer."},
    {"q": "What does LoRA freeze and what does it train?",
     "ref": "LoRA freezes the pre-trained model weights and trains small low-rank decomposition matrices injected into the layers."},
    {"q": "How many parameters does GPT-3 have?",
     "ref": "GPT-3 has 175 billion parameters."},
    {"q": "Which paper introduced multi-head attention?",
     "ref": "The paper 'Attention Is All You Need', which introduced the Transformer."},
    {"q": "What data was LLaMA trained on?",
     "ref": "LLaMA was trained only on publicly available datasets."},
]

# ---- Questions the papers cannot answer (the bot should refuse) ----
OUT_OF_SCOPE = [
    "Who won the 2022 FIFA World Cup?",
    "What is Ollama?",
]

# 1) Run every question through the RAG pipeline
rows = []
for item in QA:
    print("Asking:", item["q"])
    answer, docs = ask(item["q"])
    rows.append({
        "user_input": item["q"],
        "response": answer,
        "retrieved_contexts": [d.page_content for d in docs],
        "reference": item["ref"],
    })
    time.sleep(2)  # be gentle with the free API rate limit

# 2) Check the refusal behaviour
print("\n--- Out-of-scope check ---")
refused = 0
for q in OUT_OF_SCOPE:
    answer, _ = ask(q)
    ok = "don't have enough information" in answer.lower()
    refused += ok
    print(f"Q: {q}\nA: {answer}\nRefused correctly: {ok}\n")
    time.sleep(2)
print(f"Refusals: {refused}/{len(OUT_OF_SCOPE)}")

# 3) Score with RAGAS
print("\nRunning RAGAS (takes a few minutes)...")
dataset = EvaluationDataset.from_list(rows)
result = evaluate(
    dataset,
    metrics=[Faithfulness(), ResponseRelevancy(strictness=1)],
    llm=LangchainLLMWrapper(llm),
    embeddings=LangchainEmbeddingsWrapper(embeddings),
    run_config=RunConfig(max_workers=2, timeout=300),
)

df = result.to_pandas()

print("\n=== AVERAGE SCORES ===")
print(df.select_dtypes("number").mean().round(3))

# Save results (if the file is open in Excel, fall back to a new name)
import datetime
out = BASE_DIR / "eval_results.csv"
try:
    df.to_csv(out, index=False)
except PermissionError:
    out = BASE_DIR / f"eval_results_{datetime.datetime.now():%H%M%S}.csv"
    df.to_csv(out, index=False)
print("\nSaved:", out)