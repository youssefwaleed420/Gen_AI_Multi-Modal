import voyageai
import faiss
import numpy as np
import json
import requests

# === API and Model Setup ===
VOYAGE_API_KEY = "pa-x3-1g9PzgUhnfiBN0Bk_QswvA4thN0q7O2fVGG4Oyub"
client = voyageai.Client(api_key=VOYAGE_API_KEY)
MODEL_NAME = "voyage-multimodal-3"
OLLAMA_MODEL = "qwen3:0.6b"

# === Load FAISS index and metadata ===
index = faiss.read_index("faiss_index.idx")
with open("meta.json", "r", encoding="utf-8") as f:
    text_chunks = json.load(f)

# === Manually Set Your Question Here ===
question = "What is the Dimensions and Weight  of DMW1000  and DMW2000?"

# === Embed question using Voyage multimodal model ===
response = client.multimodal_embed(
    inputs=[[question]],
    model=MODEL_NAME,
    input_type="document"
)
query_embedding = np.array([response.embeddings[0]]).astype("float32")

# === Search the index ===
D = 1  # Number of top results to retrieve
distances, indices = index.search(query_embedding, D)
context = text_chunks[indices[0][0]]

# === Create prompt for Qwen ===
prompt = f"""Answer the following question based on the document context below:

Context:
{context}

Question: {question}
"""

# === Send to Qwen (via Ollama) ===
try:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    response_json = response.json()
    if "response" in response_json:
        print("\n🧠 LLM Answer:\n", response_json["response"].strip())
    else:
        print("\n❌ Ollama Error:\n", response_json)
except Exception as e:
    print(f"\n❌ Error connecting to Ollama: {e}")
