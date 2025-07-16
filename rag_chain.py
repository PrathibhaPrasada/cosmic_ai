# rag_chain.py

import os
from dotenv import load_dotenv
import google.generativeai as genai
from retrieval.smart_retriever import retrieve_documents
from retrieval.reranker import rerank_chunks_with_gemini

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def answer_query(query_text):
    print(f"\nQuestion: {query_text}\n")

    # Step 1: Retrieve top-k documents from Pinecone
    docs = retrieve_documents(query_text, top_k=6)
    print(f"Retrieved {len(docs)} documents from Pinecone")

    if not docs:
        print("No documents retrieved. Try re-indexing or check content.")
        return

    # 🔍 Inspect retrieved docs
    for i, doc in enumerate(docs):
        print(f"\nRetrieved Chunk {i+1}")
        print("  Score:", getattr(doc, "score", "N/A"))
        print("  Source:", doc.metadata.get("source"))
        print("  Page:", doc.metadata.get("page"))
        print("  Text Preview:", doc.metadata.get("text", "")[:200])

    # Step 2: (Optional) Rerank the documents using Gemini
    use_reranker = True  # Toggle reranker on/off for testing

    if use_reranker:
        top_docs = rerank_chunks_with_gemini(query_text, docs, top_n=3)
        print(f"\nReranked to top {len(top_docs)} most relevant chunks")
    else:
        top_docs = docs[:3]
        print("\nSkipping reranker. Using top-3 Pinecone matches.")

    if not top_docs:
        print("No documents selected after reranking.")
        return

    for i, doc in enumerate(top_docs):
        print(f"\nFinal Chunk {i+1}")
        print("  Source:", doc.metadata.get("source"))
        print("  Page:", doc.metadata.get("page"))
        print("  Text Preview:", doc.metadata.get("text", "")[:200])

    # Step 3: Build the final prompt for Gemini
    context = "\n\n".join([doc.metadata.get("text", "") for doc in top_docs])

    prompt = f"""You are a helpful assistant. Use ONLY the context below to answer the user's question.

Context:
{context}

Question:
{query_text}
"""

    # Step 4: Generate the answer using Gemini
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        print("\nAnswer:")
        print(response.text.strip())
    except Exception as e:
        print(" Gemini failed to generate a response:", e)
        return

    # Step 5: Print source metadata
    print("\n📚 Sources:")
    for doc in top_docs:
        meta = doc.metadata
        print(f"• {meta.get('source', 'unknown')} (Page {meta.get('page', '?')})")


if __name__ == "__main__":
    test_question = "Tell me about the planet Mars"
    answer_query(test_question)
