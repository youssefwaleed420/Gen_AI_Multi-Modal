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

# Constants
MODEL_NAME = "voyage-multimodal-3"
API_KEY = "pa-ckCcRTowL5eUipe_EBtlRo45D9Yyw9BtcGLe_6FNFsL"
EMBEDDINGS_FILE = "page_embeddings.json"
SAFE_DELAY = 21  # Seconds between API calls for 3 RPM

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

# Embedding
def embed_pages(pages, multimodal=True):
    inputs = [[p["text"], p["image"]] if multimodal else [p["text"]] for p in pages]
    return safe_call(vo.multimodal_embed, inputs=inputs, model=MODEL_NAME, input_type="document")

def save_embeddings(pages, result):
    data = []
    for idx, page in enumerate(pages):
        buffered = BytesIO()
        page["image"].save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data.append({
            "page_num": page["page_num"],
            "text": page["text"],
            "image_base64": img_base64,
            "embedding": result.embeddings[idx],
            "text_tokens": result.text_tokens,
            "image_pixels": result.image_pixels,
            "total_tokens": result.total_tokens
        })
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "r") as f:
            return json.load(f)
    return []

# Similarity
def compute_similarity(query_embedding, page_embeddings):
    similarities = []
    for page in page_embeddings:
        vec = np.array(page["embedding"])
        sim = np.dot(vec, query_embedding) / (np.linalg.norm(vec) * np.linalg.norm(query_embedding))
        similarities.append({**page, "similarity": sim})
    return sorted(similarities, key=lambda x: x["similarity"], reverse=True)

# Highlighting
def highlight_text_in_pdf(pdf_file, page_num, text_snippet):
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

# Find Best Snippet
def find_best_snippet(query, page_text):
    sentences = [s.strip() for s in page_text.split(". ") if len(s.strip()) > 20]
    if not sentences:
        return page_text
    result = safe_call(vo.multimodal_embed, inputs=[[s] for s in sentences], model=MODEL_NAME, input_type="document")
    query_res = safe_call(vo.multimodal_embed, inputs=[[query]], model=MODEL_NAME, input_type="query")
    if not result or not query_res:
        return "⚠️ Could not get answer due to API limits."
    q_vec = np.array(query_res.embeddings[0])
    sims = [np.dot(np.array(vec), q_vec) / (np.linalg.norm(vec) * np.linalg.norm(q_vec)) for vec in result.embeddings]
    return sentences[int(np.argmax(sims))]

# Streamlit UI
st.title("📄 PDF Multimodal Embedder with Voyage AI")

uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_pdf:
    uploaded_pdf.seek(0)
    pdf_bytes = uploaded_pdf.read()
    pdf_stream = BytesIO(pdf_bytes)
    pdf = fitz.open(stream=pdf_stream, filetype="pdf")
    total_pages = pdf.page_count
    st.info(f"PDF has {total_pages} pages.")
    pdf.close()

    selected_pages = st.multiselect(
        "Select 3 pages to embed (free tier limit)",
        options=list(range(total_pages)),
        default=[0, 1, 2]
    )
    multimodal_toggle = st.toggle("Use Multimodal Embedding (Text + Images)", value=True)

    if len(selected_pages) == 3 and st.button("🔗 Embed Selected Pages"):
        with st.spinner("Embedding pages..."):
            pages = extract_page_content(uploaded_pdf, selected_pages)
            result = embed_pages(pages, multimodal=multimodal_toggle)
            if result:
                save_embeddings(pages, result)
                st.success("✅ Embedding completed and saved")
                st.write(f"📝 Text Tokens: {result.text_tokens}")
                st.write(f"🖼️ Image Pixels: {result.image_pixels}")
                st.write(f"📦 Total Tokens: {result.total_tokens}")
            else:
                st.error("❌ Embedding failed due to API limits.")

        st.subheader("📦 Saved Embeddings")
        st.json(load_embeddings())

st.divider()

page_embeddings = load_embeddings()
if page_embeddings:
    st.subheader("❓ Ask a Question")
    query = st.text_input("Enter your question here:")
    if st.button("🔍 Search"):
        with st.spinner("Searching..."):
            query_res = safe_call(vo.multimodal_embed, inputs=[[query]], model=MODEL_NAME, input_type="query")
            if query_res:
                q_vec = np.array(query_res.embeddings[0])
                results = compute_similarity(q_vec, page_embeddings)
                top = results[0]
                best_snip = find_best_snippet(query, top["text"])
                img = highlight_text_in_pdf(uploaded_pdf, top["page_num"], best_snip)

                st.success("✅ Answer Found:")
                st.write(f"📄 **Page:** {top['page_num']}")
                st.write(f"🔗 **Similarity:** {top['similarity']:.4f}")
                st.markdown(f"### 📝 Answer:\n{best_snip}")
                st.image(img, caption=f"Highlighted Page {top['page_num']}")
            else:
                st.error("❌ Query embedding failed due to API limits.")