import os
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Setup Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.getenv("PINECONE_INDEX") or "cosmic-ai"

pc = Pinecone(api_key=PINECONE_API_KEY)

# Delete existing index to avoid duplicates and enable metadata update
if PINECONE_INDEX in pc.list_indexes().names():
    print(f"⚠️ Deleting existing index: {PINECONE_INDEX}")
    pc.delete_index(PINECONE_INDEX)

# Create new index
print(f"🆕 Creating new index: {PINECONE_INDEX} with 768 dimensions")
pc.create_index(
    name=PINECONE_INDEX,
    dimension=768,
    metric='cosine',
    spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
)

# Connect to the index
index = pc.Index(PINECONE_INDEX)

# Embedding class using Gemini
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

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Process files and collect chunks with metadata
all_chunks = []
metadata_list = []

for file in Path("text_data").glob("*.txt"):
    content = file.read_text()
    chunks = text_splitter.split_text(content)
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        metadata_list.append({
            "text": chunk,
            "source": file.name,
            "page": i
        })

print(f"📄 Loaded {len(metadata_list)} chunks from {len(list(Path('text_data').glob('*.txt')))} files.")

# Generate embeddings
embedding_model = GeminiEmbeddings()
vectors = embedding_model.embed_documents(all_chunks)

# Upsert with metadata (including filename and page)
to_upsert = [(str(i), vectors[i], metadata_list[i]) for i in range(len(vectors))]
index.upsert(vectors=to_upsert)

print(f"✅ Successfully upserted {len(vectors)} vectors to Pinecone index '{PINECONE_INDEX}'")
