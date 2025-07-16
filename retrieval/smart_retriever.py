# retrieval/smart_retriever.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from index_documents import GeminiEmbeddings  # or wherever you defined it

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX") or "cosmic-ai"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

def retrieve_documents(query_text, top_k=6):
    print(f"\nRunning query: '{query_text}'")
    embedder = GeminiEmbeddings()
    query_vector = embedder.embed_query(query_text)

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    print(f"Pinecone returned {len(results.matches)} matches")
    return results.matches
