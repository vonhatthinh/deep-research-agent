import numpy as np
from openai import OpenAI
from core.config import settings
from sklearn.metrics.pairwise import cosine_similarity
import os

# Ensure the database directory exists
if not os.path.exists('database'):
    os.makedirs('database')

client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Simple in-memory vector store
vector_store = []
next_id = 0

def get_embedding(text: str, model="text-embedding-3-small"):
   """Generates an embedding for a given text."""
   text = text.replace("\n", " ")
   return client.embeddings.create(input=[text], model=model).data[0].embedding

def add_text(text: str):
    """Adds text and its embedding to the in-memory vector store."""
    global vector_store, next_id
    if not text.strip():
        return
    embedding = get_embedding(text)
    vector_store.append({"id": next_id, "text": text, "embedding": embedding})
    next_id += 1
    print(f"INFO: Added text to vector store. Total items: {len(vector_store)}")

def query_store(query: str, top_k: int = 3) -> list[str]:
    """Queries the vector store and returns the top_k most similar text chunks."""
    if not vector_store:
        return ["Vector store is empty."]
    
    query_embedding = get_embedding(query)
    
    # Calculate similarities
    embeddings = np.array([item['embedding'] for item in vector_store])
    query_embedding_np = np.array(query_embedding).reshape(1, -1)
    
    similarities = cosine_similarity(query_embedding_np, embeddings)[0]
    
    # Get top_k results
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    
    results = [vector_store[i]['text'] for i in top_k_indices]
    return results

def clear_store():
    """Clears the in-memory vector store."""
    global vector_store, next_id
    vector_store = []
    next_id = 0
    print("INFO: Vector store cleared.")
