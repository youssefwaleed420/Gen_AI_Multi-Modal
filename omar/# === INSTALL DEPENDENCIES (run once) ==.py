# === COMPLETE DEBUGGED VERSION ===
from adb_cloud_connector import get_temp_credentials
from arango import ArangoClient
from arango.exceptions import CollectionCreateError, EdgeDefinitionCreateError
import fitz  # PyMuPDF
import voyageai 
import requests
import json
import time
import re
from tqdm import tqdm
from pyvis.network import Network
import webbrowser
import os

# Initialize Voyage client
VOYAGE_API_KEY = "pa-x3-1g9PzgUhnfiBN0Bk_QswvA4thN0q7O2fVGG4Oyub"
voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

# === ARANGODB SETUP ===
print("Initializing ArangoDB connection...")
connection = get_temp_credentials(tutorialName="LangChain")
client = ArangoClient(hosts=connection["url"])
db = client.db(connection["dbName"], connection["username"], connection["password"], verify=True)

# Initialize collections
if db.has_collection('relationships'):
    db.delete_collection('relationships')
if db.has_collection('entities'):
    db.delete_collection('entities')
if db.has_graph('pdf_knowledge_graph'):
    db.delete_graph('pdf_knowledge_graph')

entities = db.create_collection('entities')
relationships = db.create_collection('relationships', edge=True)
knowledge_graph = db.create_graph('pdf_knowledge_graph')
knowledge_graph.create_edge_definition(
    edge_collection='relationships',
    from_vertex_collections=['entities'],
    to_vertex_collections=['entities']
)
print("Graph database setup complete!")

# === PDF PROCESSING ===
def extract_and_structure_pdf(pdf_path):
    """Process PDF and build knowledge graph"""
    print(f"\nProcessing PDF: {pdf_path}")
    
    try:
        # Open PDF document (using distinct variable name)
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        
        entity_count = 0
        rel_count = 0
        
        for page_num in tqdm(range(total_pages), desc="Processing pages"):
            try:
                page = pdf_document.load_page(page_num)
                text = page.get_text()
                
                if not text.strip():
                    continue
                    
                # Clean and prepare text
                text = re.sub(r'\s+', ' ', text).strip()[:4000]  # Reduced size to prevent timeouts
                
                prompt = f"""
                Extract technical specifications and relationships from this text.
                Return ONLY JSON with this exact structure:
                {{
                    "entities": [
                        {{
                            "name": "exact_name",
                            "type": "component_type",
                            "properties": {{
                                "key1": "value1",
                                "key2": "value2"
                            }}
                        }}
                    ],
                    "relationships": [
                        {{
                            "source": "source_name",
                            "target": "target_name",
                            "type": "connection_type"
                        }}
                    ]
                }}
                
                Text:
                {text}
                """
                
                # Call Qwen with error handling
                try:
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "qwen3:0.6b",
                            "prompt": prompt,
                            "stream": False,
                            "format": "json",
                            "options": {"temperature": 0.1}
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'response' in data:
                            try:
                                # Handle potential JSON strings
                                if isinstance(data['response'], str):
                                    structured_data = json.loads(data['response'])
                                else:
                                    structured_data = data['response']
                                
                                # Process entities (using entity_doc instead of doc)
                                for entity in structured_data.get('entities', []):
                                    if not isinstance(entity, dict) or 'name' not in entity:
                                        continue
                                    
                                    # Create safe key
                                    entity_key = re.sub(r'[^a-zA-Z0-9_]', '_', entity['name'])[:128]
                                    if not entity_key:
                                        continue
                                    
                                    entity_doc = {
                                        '_key': entity_key,
                                        'name': entity['name'],
                                        'type': entity.get('type', 'component'),
                                        'properties': entity.get('properties', {})
                                    }
                                    
                                    if not entities.has(entity_key):
                                        entities.insert(entity_doc)
                                        entity_count += 1
                                    
                                # Process relationships
                                for rel in structured_data.get('relationships', []):
                                    if not all(k in rel for k in ['source', 'target']):
                                        continue
                                    
                                    source_key = re.sub(r'[^a-zA-Z0-9_]', '_', rel['source'])[:128]
                                    target_key = re.sub(r'[^a-zA-Z0-9_]', '_', rel['target'])[:128]
                                    
                                    if (entities.has(source_key) and 
                                        entities.has(target_key) and
                                        source_key != target_key):
                                        
                                        rel_doc = {
                                            '_from': f"entities/{source_key}",
                                            '_to': f"entities/{target_key}",
                                            'type': rel.get('type', 'related_to')
                                        }
                                        relationships.insert(rel_doc)
                                        rel_count += 1
                                        
                            except json.JSONDecodeError:
                                print(f"Invalid JSON on page {page_num + 1}")
                            except Exception as e:
                                print(f"Processing error on page {page_num + 1}: {str(e)}")
                                
                except requests.exceptions.RequestException as e:
                    print(f"API error on page {page_num + 1}: {str(e)}")
                    
            except Exception as e:
                print(f"Page {page_num + 1} error: {str(e)}")
                
        pdf_document.close()
        print(f"\nPDF processing complete! Extracted {entity_count} entities and {rel_count} relationships")
        return entity_count, rel_count
        
    except Exception as e:
        print(f"Failed to open PDF: {str(e)}")
        return 0, 0

# === INTERACTIVE QUERY SYSTEM ===
class InteractiveQA:
    def __init__(self, db):
        self.db = db
        self.visualization_file = "knowledge_graph.html"
        
    def visualize_graph(self, highlight_nodes=None):
        """Generate interactive visualization with highlighted nodes"""
        try:
            net = Network(notebook=True, height="750px", width="100%", cdn_resources='remote')
            
            # Add all nodes
            for entity in db.collection('entities').all():
                color = "#97c2fc"  # Default blue
                size = 15
                if highlight_nodes and entity['_key'] in highlight_nodes:
                    color = "#ff7f7f"  # Highlight color
                    size = 25
                net.add_node(
                    entity['_key'], 
                    label=entity['name'], 
                    title=f"{entity['type']}\n{json.dumps(entity.get('properties', {}), indent=2)}",
                    color=color,
                    size=size
                )
            
            # Add all edges
            for rel in db.collection('relationships').all():
                net.add_edge(
                    rel['_from'].split('/')[1],
                    rel['_to'].split('/')[1],
                    title=rel.get('type', 'related')
                )
            
            net.show(self.visualization_file)
            return os.path.abspath(self.visualization_file)
            
        except Exception as e:
            print(f"Visualization error: {str(e)}")
            return None
    
    def query(self, question):
        """Process user question and return answer with visualization"""
        try:
            # First try direct property matches
            cursor = self.db.aql.execute(
                """
                FOR entity IN entities
                    FILTER entity.properties != {} 
                    AND (
                        LIKE(entity.name, @query, true)
                        OR LIKE(entity.properties, @query, true)
                        OR LIKE(entity.type, @query, true)
                    )
                    LIMIT 10
                    RETURN entity
                """,
                bind_vars={'query': f"%{question}%"}
            )
            results = list(cursor)
            
            if not results:
                return "I couldn't find that information in the document.", None
            
            # Build context and track relevant nodes
            context = "Relevant Information:\n"
            highlight_nodes = set()
            for entity in results:
                highlight_nodes.add(entity['_key'])
                context += f"\n- {entity['name']} ({entity['type']}):\n"
                for k, v in entity.get('properties', {}).items():
                    context += f"  • {k}: {v}\n"
            
            # Get relationships
            cursor = self.db.aql.execute(
                """
                FOR v, e IN 1..2 ANY @entities GRAPH 'pdf_knowledge_graph'
                    RETURN {vertex: v, edge: e}
                """,
                bind_vars={'entities': [f"entities/{e['_key']}" for e in results]}
            )
            
            rels = list(cursor)
            if rels:
                context += "\nRelationships:\n"
                for rel in rels:
                    if 'edge' in rel:
                        src = rel['edge']['_from'].split('/')[1]
                        tgt = rel['edge']['_to'].split('/')[1]
                        highlight_nodes.update([src, tgt])
                        context += f"- {src} → {tgt} ({rel['edge']['type']})\n"
            
            # Generate answer
            prompt = f"""Technical Document Assistant:
            
            Context:
            {context}
            
            Question: {question}
            
            Provide a concise answer using ONLY the context above.
            If the information isn't available, say "Not specified in the document."
            """
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen3:0.6b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=30
            )
            
            answer = response.json().get('response', 'Error processing response').strip()
            viz_path = self.visualize_graph(highlight_nodes)
            return answer, viz_path
            
        except Exception as e:
            return f"Error processing query: {str(e)}", None

# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Process PDF and build graph
    pdf_path = r"C:\Users\STW\Downloads\MA_DWM1000_2000_en_120509.pdf"
    entity_count, rel_count = extract_and_structure_pdf(pdf_path)
    
    if entity_count > 0:
        # Initialize interactive system
        qa = InteractiveQA(db)
        print("\nInitial visualization generated...")
        initial_viz = qa.visualize_graph()
        if initial_viz:
            webbrowser.open(f'file://{initial_viz}')
        
        # Interactive question loop
        print("\nEnter your questions about the document (type 'exit' to quit):")
        while True:
            question = input("\nYour question: ").strip()
            if question.lower() in ['exit', 'quit']:
                break
                
            if not question:
                continue
                
            answer, viz_file = qa.query(question)
            print(f"\nAnswer: {answer}")
            
            if viz_file:
                webbrowser.open(f'file://{viz_file}')
                print("Updated visualization opened in browser")
            else:
                print("No visualization update available")
        
        print("\nSession ended. Final visualization saved in knowledge_graph.html")
    else:
        print("Failed to extract information from PDF. Please check the file path and content.")