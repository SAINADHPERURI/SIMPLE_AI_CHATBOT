print("APP.PY STARTED")

import os
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ---------------------------------
# Configure Gemini (TEXT GENERATION)
# ---------------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "models/gemini-2.5-flash"
print("Using Gemini model:", MODEL_NAME)

llm = genai.GenerativeModel(MODEL_NAME)

# ---------------------------------
# Load Local Embedding Model
# ---------------------------------
print("Loading local embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------
# Load PDF
# ---------------------------------
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ---------------------------------
# Chunk Text
# ---------------------------------
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

# ---------------------------------
# Create Embedding (LOCAL)
# ---------------------------------
def get_embedding(text):
    return embedder.encode(text).astype("float32")

# ---------------------------------
# Build FAISS Index
# ---------------------------------
def build_index(chunks):
    embeddings = []

    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i + 1}/{len(chunks)}")
        embeddings.append(get_embedding(chunk))

    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index

# ---------------------------------
# Retrieve Relevant Chunks
# ---------------------------------
def retrieve(query, index, chunks, k=3):
    query_embedding = get_embedding(query)
    _, indices = index.search(np.array([query_embedding]), k)
    return [chunks[i] for i in indices[0]]

# ---------------------------------
# Generate Answer Using Gemini
# ---------------------------------
def generate_answer(query, context):
    prompt = f"""
Answer ONLY from the context below.
If the answer is not present, say "Answer not found in documents."

Context:
{context}

Question:
{query}
"""
    response = llm.generate_content(prompt)
    return response.text

# ---------------------------------
# MAIN
# ---------------------------------
if __name__ == "__main__":
    print("Starting Naive RAG Application...")

    print("Loading PDF...")
    text = load_pdf("documents/notes.pdf")
    print("PDF loaded successfully")

    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Total chunks created: {len(chunks)}")

    print("Creating FAISS index...")
    index = build_index(chunks)

    print("Naive RAG system ready!!")

    while True:
        query = input("\nEnter your question (or type exit): ")
        if query.lower() == "exit":
            print("Exiting...")
            break

        context = "\n".join(retrieve(query, index, chunks))
        answer = generate_answer(query, context)

        print("\nAnswer:\n", answer)