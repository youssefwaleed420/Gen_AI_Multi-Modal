import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import json
import os
import time
from io import BytesIO
from voyageai import Client, error
import base64
import faiss
import requests

# Constants
MODEL_NAME = "voyage-multimodal-3"
API_KEY = "pa-ckCcRTowL5eUipe_EBtlRo45D9Yyw9BtcGLe_6FNFsL"
SAFE_DELAY = 10  # Slightly reduced delay because we chunk requests
FAISS_INDEX_FILE = "faiss_index.idx"
META_FILE = "meta.json"
OLLAMA_MODEL = "llama3.2:latest"  # Your local Ollama model

# Init Voyage client
vo = Client(api_key=API_KEY)

# Helper: Rate-limited API call
def safe_call(api_func, *args, **kwargs):
    try:
        result = api_func(*args, **kwargs)
        time.sleep(SAFE_DELAY)  # Wait to avoid hitting RPM
        return result
    except error.RateLimitError:
        st.warning("⚠️ Rate limit hit. Waiting 30s before retry...")
        time.sleep(30)
        return safe_call(api_func, *args, **kwargs)
    except Exception as e:
        st.error(f"❌ API Error: {e}")
        return None

# PDF Extraction
def extract_page_content(pdf_file, page_numbers):
    pdf_file.seek(0)
    pdf_bytes = pdf_file.read()
    pdf_stream = BytesIO(pdf_bytes)
    pdf = fitz.open(stream=pdf_stream, filetype="pdf")
    pages = []
    for page_num in page_numbers:
        if page_num < pdf.page_count:
            page = pdf.load_page(page_num)
            text = page.get_text("text")
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append({"page_num": page_num, "text": text, "image": img})
    pdf.close()
    return pages

# Embedding and save to FAISS
def embed_and_save_to_faiss(pages, multimodal=True):
    inputs = [[p["text"], p["image"]] if multimodal else [p["text"]] for p in pages]
    result = safe_call(vo.multimodal_embed, inputs=inputs, model=MODEL_NAME, input_type="document")
    if not result:
        return None

    # Convert to numpy array
    embeddings = np.array(result.embeddings).astype("float32")
    dim = embeddings.shape[1]

    # Create FAISS index
    if os.path.exists(FAISS_INDEX_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)

    # Save metadata
    metadata = []
    for idx, page in enumerate(pages):
        buffered = BytesIO()
        page["image"].save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        meta_entry = {
            "page_num": page["page_num"],
            "text": page["text"],
            "image_base64": img_base64
        }
        metadata.append(meta_entry)

        # Show metadata for each page
        with st.expander(f"📄 Page {page['page_num']} Metadata"):
            st.markdown(f"🔹 **Text Snippet:**\n{page['text'][:500]}{'...' if len(page['text']) > 500 else ''}")

    # Append or save metadata
    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            existing_meta = json.load(f)
        metadata = existing_meta + metadata
    with open(META_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    # Show total token info
    st.info(f"📝 **Total Text Tokens:** {result.text_tokens}")
    if multimodal and hasattr(result, "image_pixels"):
        st.info(f"🖼️ **Total Image Pixels (approx):** {result.image_pixels}")
    st.success(f"📦 **Grand Total Tokens (Text + Images):** {result.total_tokens}")

    return result


# Search FAISS
def search_faiss(query_embedding, k=1):
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(META_FILE):
        st.error("❌ FAISS index or metadata not found.")
        return None, None

    index = faiss.read_index(FAISS_INDEX_FILE)
    query_vec = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query_vec, k)

    with open(META_FILE, "r") as f:
        metadata = json.load(f)

    return metadata[indices[0][0]], distances[0][0]

# Find best snippet in page text
def find_best_snippet(query, page_text):
    sentences = [s.strip() for s in page_text.split(". ") if len(s.strip()) > 20]
    if not sentences:
        return page_text
    result = safe_call(vo.multimodal_embed, inputs=[[s] for s in sentences], model=MODEL_NAME, input_type="document")
    query_res = safe_call(vo.multimodal_embed, inputs=[[query]], model=MODEL_NAME, input_type="query")
    if not result or not query_res:
        return None
    q_vec = np.array(query_res.embeddings[0])
    sims = [np.dot(np.array(vec), q_vec) / (np.linalg.norm(vec) * np.linalg.norm(q_vec)) for vec in result.embeddings]
    return sentences[int(np.argmax(sims))]

# Highlight text snippet in PDF
def highlight_pdf_snippet(pdf_file, page_num, text_snippet):
    pdf_file.seek(0)
    pdf_bytes = pdf_file.read()
    pdf_stream = BytesIO(pdf_bytes)
    pdf = fitz.open(stream=pdf_stream, filetype="pdf")
    page = pdf.load_page(page_num)
    for rect in page.search_for(text_snippet):
        page.add_highlight_annot(rect).update()
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    pdf.close()
    return img

# Ask Ollama
def ask_ollama(context, question):
    prompt = f"""Answer the question based on the document context below:
Context:
{context}

Question: {question}
"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    data = response.json()
    return data.get("response", "⚠️ Ollama failed to generate a response.")

# Streamlit UI
st.title("📄 PDF Multimodal Embedder with Highlighting (FAISS + VoyageAI)")

uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
multimodal_toggle = st.toggle("Use Multimodal Embedding (Text + Images)", value=True)

if uploaded_pdf:
    uploaded_pdf.seek(0)
    pdf_bytes = uploaded_pdf.read()
    pdf_stream = BytesIO(pdf_bytes)
    pdf = fitz.open(stream=pdf_stream, filetype="pdf")
    total_pages = pdf.page_count
    st.info(f"PDF has {total_pages} pages.")
    pdf.close()

    embed_mode = st.radio("Choose embedding mode:", ["Select up to 3 pages", "Embed entire PDF (rate-limited)"])

    if embed_mode == "Select up to 3 pages":
        selected_pages = st.multiselect(
            "Select up to 3 pages",
            options=list(range(total_pages)),
            default=list(range(min(3, total_pages)))
        )
        if 0 < len(selected_pages) <= 3 and st.button("🔗 Embed Selected Pages"):
            with st.spinner("Embedding selected pages..."):
                pages = extract_page_content(uploaded_pdf, selected_pages)
                result = embed_and_save_to_faiss(pages, multimodal=multimodal_toggle)
                if result:
                    st.success("✅ Selected pages embedded and saved.")
                else:
                    st.error("❌ Embedding failed.")
    else:
        if st.button("🔗 Embed Entire PDF (Rate Limited)"):
            all_page_numbers = list(range(total_pages))
            st.info("⚡ Embedding entire PDF in chunks of 3 pages per 30s...")
            for i in range(0, len(all_page_numbers), 3):
                chunk_pages = all_page_numbers[i:i+3]
                with st.spinner(f"Embedding pages {chunk_pages}..."):
                    pages = extract_page_content(uploaded_pdf, chunk_pages)
                    result = embed_and_save_to_faiss(pages, multimodal=multimodal_toggle)
                    if result:
                        st.success(f"✅ Pages {chunk_pages} embedded.")
                    else:
                        st.error(f"❌ Failed to embed pages {chunk_pages}.")
                if i + 3 < len(all_page_numbers):
                    st.info("⏳ Waiting 30s to avoid rate limits...")
                    time.sleep(30)

st.divider()

# Search UI
st.subheader("❓ Ask a Question")
query = st.text_input("Enter your question here:")
if st.button("🔍 Search and Answer"):
    with st.spinner("Embedding query and searching FAISS..."):
        query_res = safe_call(vo.multimodal_embed, inputs=[[query]], model=MODEL_NAME, input_type="query")
        if query_res:
            best_meta, distance = search_faiss(query_res.embeddings[0])
            if best_meta:
                best_snip = find_best_snippet(query, best_meta['text'])
                if best_snip:
                    highlighted_img = highlight_pdf_snippet(uploaded_pdf, best_meta["page_num"], best_snip)
                    st.success("✅ Best Match Found")
                    st.write(f"📄 **Page:** {best_meta['page_num']}")
                    st.write(f"📊 **Distance (FAISS):** {distance:.4f}")
                    st.markdown(f"### 📝 Answer Snippet:\n{best_snip}")
                    st.image(highlighted_img, caption=f"Highlighted Page {best_meta['page_num']}")
                    answer = ask_ollama(best_meta['text'], query)
                    st.markdown(f"### 🤖 LLM Answer:\n{answer}")
                else:
                    st.error("❌ Could not find a snippet in text.")
            else:
                st.error("❌ No match found in FAISS.")
        else:
            st.error("❌ Query embedding failed.")
