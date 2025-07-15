import os
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai
from langchain.embeddings.base import Embeddings

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Setup Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX") or "cosmic-ai"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            response = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(response["embedding"])
        return embeddings

    def embed_query(self, text):
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        return response["embedding"]

def query_pinecone(query_text, top_k=5):
    embedding_model = GeminiEmbeddings()
    query_vector = embedding_model.embed_query(query_text)

    # Query with metadata included
    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    print(f"\nTop {top_k} matches for your query: '{query_text}':\n")
    
    # If result.matches is a list of matches
    for match in result.matches:
        print(f"Score: {match.score:.4f}")
        print(f"Text: {match.metadata.get('text', 'No text found')}")
        print("-" * 80)

if __name__ == "__main__":
    test_query =  "Explain the phases of the moon"
    query_pinecone(test_query)
