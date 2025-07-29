import voyageai
import faiss
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import io
import requests
from neo4j import GraphDatabase
from typing import List, Dict, Union, Optional

# Configuration
VOYAGE_API_KEY = "pa-tDh9PAJmIfaPahq1-GkuSk8uVNGrI69sq3uxpiGK8Y7"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"
PDF_PATH = r"C:\Users\STW\Downloads\MA_DWM1000_2000_en_120509.pdf"

# Neo4j Connection - Using simplified syntax for local Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "omarnasser")

# Initialize Voyage AI client
voyageai.api_key = VOYAGE_API_KEY
client = voyageai.Client()

class HybridSearchSystem:
    def __init__(self):
        self.faiss_index = None
        self.metadata = []
        self.neo4j_driver = None
        self.vector_index_created = False
        
    def connect_neo4j(self):
        try:
            self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
            
            # Verify connection with a simple query
            with self.neo4j_driver.session() as session:
                result = session.run("RETURN 1 AS test")
                if result.single()["test"] == 1:
                    print("✅ Successfully connected to Neo4j")
                    
            # Create vector index
            self.create_neo4j_vector_index()
            
        except Exception as e:
            print(f"⚠️ Neo4j connection failed: {str(e)}")
            self.neo4j_driver = None

    def create_neo4j_vector_index(self):
        """Create vector index in Neo4j using correct syntax"""
        if not self.neo4j_driver:
            return
            
        try:
            with self.neo4j_driver.session() as session:
                # First check if index exists
                index_exists = session.run(
                    "SHOW INDEXES WHERE type = 'VECTOR' AND name = 'document_embeddings'"
                ).single()
                
                if not index_exists:
                    # Create index with correct syntax for Neo4j 5.x
                    session.run("""
                    CREATE VECTOR INDEX document_embeddings 
                    FOR (d:Document) ON (d.embedding)
                    OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1024,
                        `vector.similarity_function`: 'cosine'
                    }
                    }
                    """)

                    print("✅ Created vector index in Neo4j")
                    self.vector_index_created = True
                else:
                    print("ℹ️ Vector index already exists")
                    self.vector_index_created = True
                    
        except Exception as e:
            print(f"⚠️ Failed to create vector index: {str(e)}")
            self.vector_index_created = False

    def process_pdf(self, pdf_path: str) -> List[Union[str, Image.Image]]:
        """Extract text and first image from PDF"""
        doc = fitz.open(pdf_path)
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
        return [pdf_text, pdf_image] if pdf_image else [pdf_text]

    def create_faiss_index(self, documents: List[List[Union[str, Image.Image]]]):
        """Create FAISS index from documents"""
        response = client.multimodal_embed(
            inputs=documents,
            model="voyage-multimodal-3",
            input_type="document"
        )
        
        embeddings = np.array(response.embeddings).astype("float32")
        dimension = embeddings.shape[1]
        
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)
        self.metadata = [doc[0] for doc in documents]
        
        print(f"✅ Created FAISS index with {len(embeddings)} vectors")
        return response

    def store_in_neo4j(self, documents: List[List[Union[str, Image.Image]]], embeddings: List[List[float]]):
        """Store documents and embeddings in Neo4j"""
        if not self.neo4j_driver or not self.vector_index_created:
            print("⚠️ Neo4j not properly configured - skipping graph storage")
            return
            
        try:
            with self.neo4j_driver.session() as session:
                for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                    session.run(
                        """
                        MERGE (d:Document {id: $id})
                        SET d.text = $text,
                            d.embedding = $embedding,
                            d.has_image = $has_image,
                            d.source = $source
                        """,
                        {
                            "id": f"doc_{i}",
                            "text": doc[0],
                            "embedding": embedding,
                            "has_image": len(doc) > 1,
                            "source": PDF_PATH
                        }
                    )
            print(f"💾 Stored {len(documents)} documents in Neo4j")
        except Exception as e:
            print(f"⚠️ Failed to store in Neo4j: {str(e)}")

    def search(self, query: str, top_k: int = 3) -> Dict:
        """Search using both FAISS and Neo4j"""
        # Embed the question
        question_embedding = client.multimodal_embed(
            inputs=[[query]],
            model="voyage-multimodal-3",
            input_type="document"
        ).embeddings[0]
        
        # FAISS search
        faiss_results = []
        if self.faiss_index:
            query_embedding = np.array([question_embedding]).astype("float32")
            distances, indices = self.faiss_index.search(query_embedding, top_k)
            
            faiss_results = [
                {
                    "text": self.metadata[idx],
                    "score": float(1 / (1 + distances[0][i]))
                }
                for i, idx in enumerate(indices[0])
            ]
        
        # Neo4j vector search (if available)
        neo4j_results = []
        if self.neo4j_driver and self.vector_index_created:
            try:
                with self.neo4j_driver.session() as session:
                    result = session.run(
                        """
                        CALL db.index.vector.queryNodes('document_embeddings', $top_k, $embedding)
                        YIELD node, score
                        RETURN node.text AS text, score
                        ORDER BY score DESC
                        """,
                        {"top_k": top_k, "embedding": question_embedding}
                    )
                    neo4j_results = [dict(record) for record in result]
            except Exception as e:
                print(f"⚠️ Neo4j search failed: {str(e)}")
        
        return {
            "faiss_results": faiss_results,
            "neo4j_results": neo4j_results
        }

    def query_llm(self, prompt: str) -> str:
        """Query Ollama LLM with error handling"""
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("response", "No response from LLM")
        except Exception as e:
            print(f"⚠️ LLM query failed: {str(e)}")
            return f"Error generating answer: {str(e)}"

def main():
    search_system = HybridSearchSystem()
    search_system.connect_neo4j()
    
    # Process PDF
    document = search_system.process_pdf(PDF_PATH)
    documents = [document]
    
    # Create FAISS index
    embed_response = search_system.create_faiss_index(documents)
    
    # Store in Neo4j
    search_system.store_in_neo4j(documents, embed_response.embeddings)
    
    # Example query
    question = "What is The DWM electromagnetic flow meters and switches are designed to?"
    print(f"\n❓ Question: {question}")
    
    # Search both indexes
    results = search_system.search(question)
    
    # Combine results
    context = results["faiss_results"] or results["neo4j_results"]
    
    if context:
        context_text = "\n\n".join([
            f"Document (Score: {res['score']:.2f}):\n{res['text']}" 
            for res in context
        ])
        
        prompt = f"""Answer the question based on these documents:
        
        {context_text}
        
        Question: {question}
        Provide a detailed technical answer with specifications from the document:"""
        
        answer = search_system.query_llm(prompt)
        print("\n💡 Answer:")
        print(answer.strip())
    else:
        print("⚠️ No relevant documents found")

if __name__ == "__main__":
    try:
        import faiss
        main()
    except ImportError:
        print("Error: FAISS not installed. Install with:")
        print("conda install -c conda-forge faiss-cpu")