# retrieval/reranker.py

import google.generativeai as genai

def rerank_chunks_with_gemini(query, documents, top_n=3):
    """
    Re-ranks retrieved chunks using Gemini and returns top_n most relevant ones.
    """
    if not documents:
        return []

    # Build the reranking prompt
    prompt = f"""You are a helpful AI assistant.
A user asked: "{query}"

Below are some document chunks. Rank them from most to least relevant.
Just return the numbers of the most relevant chunks (comma-separated).

"""

    for i, doc in enumerate(documents):
        snippet = doc['metadata']['text'].replace("\n", " ")
        prompt += f"[{i+1}] {snippet}\n"

    prompt += "\nReply ONLY with numbers, like: 2, 1, 4"

    # Call Gemini
    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    response = model.generate_content(prompt)

    try:
        ranking = [int(i.strip()) for i in response.text.split(",")]
        reranked = [documents[i - 1] for i in ranking if 0 < i <= len(documents)]
        return reranked[:top_n]
    except Exception as e:
        print("Gemini reranking failed. Returning original order. Error:", e)
        return documents[:top_n]
