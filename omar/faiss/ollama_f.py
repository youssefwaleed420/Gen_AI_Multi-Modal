import voyageai
import faiss
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import io
import json
import os
import requests

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# === Set your API key ===
VOYAGE_API_KEY = "pa-x3-1g9PzgUhnfiBN0Bk_QswvA4thN0q7O2fVGG4Oyub"  # Replace with your actual key
client = voyageai.Client(api_key=VOYAGE_API_KEY)

# === Load PDF (from path) ===
pdf_path = r"C:\Users\STW\Downloads\MA_DWM1000_2000_en_120509.pdf"
doc = fitz.open(pdf_path)

# === Extract text and first image ===
pdf_text = ""
pdf_image = None

for page in doc:
    pdf_text += page.get_text()

    if not pdf_image:
        image_list = page.get_images(full=True)
        if image_list:
            xref = image_list[0][0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pdf_image = Image.open(io.BytesIO(image_bytes)).resize((256, 256))

doc.close()

# === Prepare multimodal documents ===
documents = [[pdf_text, pdf_image]] if pdf_image else [[pdf_text]]

# === Embed with Voyage multimodal model ===
MODEL_NAME = "voyage-multimodal-3"
response = client.multimodal_embed(
    inputs=documents,
    model=MODEL_NAME,
    input_type="document"
)

# === Convert to numpy for FAISS ===
embeddings = np.array(response.embeddings).astype("float32")

# === Create and Save FAISS Index ===
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, "faiss_index.idx")

# === Save metadata ===
with open("meta.json", "w", encoding="utf-8") as f:
    json.dump([pdf_text], f)  # saving raw text for context

# === Print Info ===
print(f"\n✅ Number of vectors: {len(response.embeddings)}")
print(f"📝 Text tokens: {response.text_tokens}")
print(f"🖼️ Image pixels: {response.image_pixels}")
print(f"📊 Total tokens: {response.total_tokens}")

# === Step 1: Ask your question ===
question = "What is the operating voltage of the DWM2000 module?"

# === Step 2: Embed the question ===
question_embedding = client.multimodal_embed(
    inputs=[[question]],
    model=MODEL_NAME,
    input_type="document"
).embeddings[0]

# === Step 3: Search FAISS index ===
D = 1  # number of nearest neighbors to return
question_embedding = np.array([question_embedding]).astype("float32")
distances, indices = index.search(question_embedding, D)

# === Step 4: Load the matched context ===
with open("meta.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)
context = metadata[indices[0][0]]

# === Step 5: Create prompt ===
prompt = f"""Answer the following question based on the document context below:

Context:
{context}

Question: {question}
"""

# === Step 6: Send to Qwen via Ollama ===
OLLAMA_MODEL = "qwen3:0.6b"  # Ensure it's pulled: `ollama pull qwen:0.5b`

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
)

# === Step 7: Show Answer ===
response_json = response.json()

if "response" in response_json:
    generated_text = response_json["response"]
    print("\n🧠 LLM Answer:\n", generated_text.strip())
else:
    print("\n❌ Ollama Error:\n", response_json)

