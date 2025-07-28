import streamlit as st
import numpy as np
import json
import os
from io import BytesIO
from voyageai import Client
import base64
import faiss
import pymupdf
from PIL import Image
import requests

# ============================
# Constants & Config
# ============================
MODEL_NAME = "voyage-multimodal-3"
FAISS_INDEX_FILE = "faiss_index.idx"
META_FILE = "meta.json"
PDF_STORAGE_DIR = "pdf_storage"
VOYAGE_API_KEY = "pa-ckCcRTowL5eUipe_EBtlRo45D9Yyw9BtcGLe_6FNFsL"  # Your Voyage API key
GEMINI_API_KEY = "AIzaSyAvyufMgHqXW2DF88kpMvKnRmqX9QgvviA"  # Your Gemini API key
OLLAMA_MODEL = "llama3.2:latest"   # Change if using llama2, llama3.1, etc.
OLLAMA_URL = "http://localhost:11434/api/generate"

# Init Voyage client
vo = Client(api_key=VOYAGE_API_KEY)

# Create storage directory
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

# ============================
# Utility Functions
# ============================

def reset_data():
    """Reset all stored embeddings and PDFs"""
    for file in [FAISS_INDEX_FILE, META_FILE]:
        if os.path.exists(file):
            os.remove(file)
    for file in os.listdir(PDF_STORAGE_DIR):
        os.remove(os.path.join(PDF_STORAGE_DIR, file))
    st.warning("🗑️ Data reset complete.")

def safe_call(func, *args, **kwargs):
    """Call an API safely with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

def extract_pages(pdf_file, page_numbers):
    """Extract text and images from PDF pages"""
    pdf_file.seek(0)
    pdf = pymupdf.open(stream=BytesIO(pdf_file.read()), filetype="pdf")
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

def embed_pages(pages, multimodal=True, pdf_filename=None):
    """Embed PDF pages and save into FAISS"""
    inputs = [[p["text"], p["image"]] if multimodal else [p["text"]] for p in pages]
    result = safe_call(vo.multimodal_embed, inputs=inputs, model=MODEL_NAME, input_type="document")
    if not result:
        return None

    embeddings = np.array(result.embeddings).astype("float32")

    # Load or create FAISS HNSW index
    if os.path.exists(FAISS_INDEX_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
    else:
        index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
        index.hnsw.efConstruction = 64

    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)

    # Save metadata
    metadata = []
    for page in pages:
        buffered = BytesIO()
        page["image"].save(buffered, format="PNG")
        metadata.append({
            "page_num": page["page_num"],
            "text": page["text"],
            "image_base64": base64.b64encode(buffered.getvalue()).decode("utf-8"),
            "pdf_filename": pdf_filename
        })

    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            existing_meta = json.load(f)
        metadata = existing_meta + metadata

    with open(META_FILE, "w") as f:
        json.dump(metadata, f)

    st.success(f"✅ Embedded {len(pages)} pages. Tokens: {result.total_tokens}, Pixels: {result.image_pixels}")
    return result

def search_multiple(query_embedding, k=5):
    """Search FAISS for multiple results"""
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(META_FILE):
        return []

    index = faiss.read_index(FAISS_INDEX_FILE)
    query_vec = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query_vec, min(k, index.ntotal))

    with open(META_FILE, "r") as f:
        metadata = json.load(f)

    results = []
    seen_pages = set()
    for i, idx in enumerate(indices[0]):
        if distances[0][i] <= 2.0:  # Relevance threshold
            page_key = (metadata[idx].get('pdf_filename', ''), metadata[idx]['page_num'])
            if page_key not in seen_pages:
                seen_pages.add(page_key)
                results.append({'metadata': metadata[idx], 'distance': distances[0][i]})

    return results

def find_snippet(query, text, k=1):
    """Find best snippet using FAISS IndexHNSWFlat"""
    sentences = [s.strip() for s in text.split(". ") if len(s.strip()) > 20]
    if not sentences:
        return text[:500]

    result = safe_call(vo.multimodal_embed, inputs=[[s] for s in sentences], model=MODEL_NAME, input_type="document")
    query_res = safe_call(vo.multimodal_embed, inputs=[[query]], model=MODEL_NAME, input_type="query")

    if not result or not query_res:
        return sentences[0]

    sent_embs = np.array(result.embeddings).astype("float32")
    q_vec = np.array([query_res.embeddings[0]]).astype("float32")

    dim = sent_embs.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efSearch = 64
    index.add(sent_embs)

    distances, indices = index.search(q_vec, min(k, len(sentences)))
    best_snippets = [sentences[i] for i in indices[0]]
    return best_snippets[0] if k == 1 else best_snippets

def highlight_pdf(pdf_source, page_num, text_snippet):
    """Highlight matching text snippet in PDF"""
    try:
        if isinstance(pdf_source, str):
            with open(pdf_source, "rb") as f:
                pdf_bytes = f.read()
        else:
            pdf_source.seek(0)
            pdf_bytes = pdf_source.read()

        pdf = pymupdf.open(stream=BytesIO(pdf_bytes), filetype="pdf")
        if page_num >= pdf.page_count:
            return None

        page = pdf.load_page(page_num)

        import re
        clean_text = re.sub(r'\s+', ' ', text_snippet.strip())
        rects = page.search_for(clean_text[:100] if len(clean_text) > 100 else clean_text)

        if rects:
            for rect in rects:
                page.add_highlight_annot(rect).update()

        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pdf.close()
        return img
    except:
        return None

def ask_ollama(contexts_and_pages, question):
    """Generate answer using Ollama"""
    try:
        context_text = ""
        for context, page_info in contexts_and_pages:
            pdf_name = page_info.get('pdf_filename', 'PDF')
            page_num = page_info.get('page_num', '?')
            context_text += f"\n--- {pdf_name}, Page {page_num} ---\n{context}\n"

        prompt = f"""Answer the question based on these contexts:
{context_text}

Question: {question}

Provide a comprehensive answer, referencing specific pages when relevant.
"""

        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        )

        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"❌ Ollama error: {response.text}"
    except Exception as e:
        return f"❌ Ollama exception: {e}"

# ============================
# Streamlit UI
# ============================

st.title("📄 Multi-Page PDF Search with LLaMA via Ollama")

if st.button("🗑️ Reset Data"):
    reset_data()

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
multimodal = st.toggle("Multimodal Embedding", value=True)

if uploaded_pdf:
    pdf = pymupdf.open(stream=BytesIO(uploaded_pdf.read()), filetype="pdf")
    total_pages = pdf.page_count
    st.info(f"PDF has {total_pages} pages")
    pdf.close()

    embed_mode = st.radio("Mode:", ["Select pages (max 3)", "Embed all"])

    if embed_mode == "Select pages (max 3)":
        selected = st.multiselect("Pages:", list(range(total_pages)), default=list(range(min(3, total_pages))))
        if 0 < len(selected) <= 3 and st.button("🔗 Embed Selected"):
            pdf_path = os.path.join(PDF_STORAGE_DIR, uploaded_pdf.name)
            uploaded_pdf.seek(0)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.read())
            pages = extract_pages(uploaded_pdf, selected)
            embed_pages(pages, multimodal, uploaded_pdf.name)
    else:
        if st.button("🔗 Embed All"):
            pdf_path = os.path.join(PDF_STORAGE_DIR, uploaded_pdf.name)
            uploaded_pdf.seek(0)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.read())
            pages = extract_pages(uploaded_pdf, list(range(total_pages)))
            embed_pages(pages, multimodal, uploaded_pdf.name)

st.divider()

st.subheader("❓ Search")
query = st.text_input("Question:")

if st.button("🔍 Search") and query.strip():
    if not os.path.exists(FAISS_INDEX_FILE):
        st.error("❌ No embedded documents found")
    else:
        query_res = safe_call(vo.multimodal_embed, inputs=[[query]], model=MODEL_NAME, input_type="query")
        if query_res:
            results = search_multiple(query_res.embeddings[0])
            if results:
                st.success(f"✅ Found {len(results)} relevant pages")

                contexts = []
                for i, result in enumerate(results):
                    meta = result['metadata']
                    distance = result['distance']

                    with st.expander(f"📄 Page {meta['page_num']} (Distance: {distance:.3f})", expanded=(i == 0)):
                        snippet = find_snippet(query, meta['text'])
                        st.markdown(f"**Snippet:** {snippet}")
                        contexts.append((meta['text'], meta))

                        pdf_to_use = None
                        if meta.get('pdf_filename'):
                            pdf_path = os.path.join(PDF_STORAGE_DIR, meta['pdf_filename'])
                            if os.path.exists(pdf_path):
                                pdf_to_use = pdf_path
                        if not pdf_to_use and uploaded_pdf and uploaded_pdf.name == meta.get('pdf_filename'):
                            pdf_to_use = uploaded_pdf

                        if pdf_to_use:
                            highlighted = highlight_pdf(pdf_to_use, meta["page_num"], snippet)
                            if highlighted:
                                st.image(highlighted, caption=f"Page {meta['page_num']}")
                            else:
                                img_data = base64.b64decode(meta["image_base64"])
                                st.image(Image.open(BytesIO(img_data)), caption=f"Page {meta['page_num']}")
                        else:
                            img_data = base64.b64decode(meta["image_base64"])
                            st.image(Image.open(BytesIO(img_data)), caption=f"Page {meta['page_num']}")

                if contexts:
                    with st.spinner("Generating answer with LLaMA..."):
                        answer = ask_ollama(contexts, query)
                        st.markdown("### 🤖 Answer:")
                        st.markdown(answer)
            else:
                st.warning("⚠️ No relevant results found")
