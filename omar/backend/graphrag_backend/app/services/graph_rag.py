import voyageai
import faiss
import numpy as np
from neo4j import GraphDatabase
import requests
import json
from typing import List, Dict, Union, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from .database_manager import DatabaseManager
from graphrag_backend.app.services.ollama_llm import AdvancedOllamaLLM


class AdvancedGraphRAGSystem:
    """Enhanced GraphRAG system with advanced Voyage AI integration"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.faiss_index = None
        self.metadata = []
        self.neo4j_driver = None
        self.vector_index_created = False
        self.knowledge_graph_created = False
        self.db_manager = db_manager
        self.llm = AdvancedOllamaLLM()
        self.voyage_embeddings_cache = {}
        self.graph_traversal_history = defaultdict(list)
        
    def connect_neo4j(self):
        """Enhanced Neo4j connection with GraphRAG optimizations"""
        try:
            self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
            
            with self.neo4j_driver.session() as session:
                result = session.run("RETURN 1 AS test")
                if result.single()["test"] == 1:
                    logger.info("✅ Advanced Neo4j connection established")
                    
            self.create_advanced_graph_schema()
            
        except Exception as e:
            logger.error(f"⚠️ Neo4j connection failed: {str(e)}")
            self.neo4j_driver = None

    def create_advanced_graph_schema(self):
        """Create advanced GraphRAG schema with relationship weights"""
        if not self.neo4j_driver:
            return
            
        try:
            with self.neo4j_driver.session() as session:
                # Enhanced constraints
                session.run("CREATE CONSTRAINT unique_concept IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")
                session.run("CREATE CONSTRAINT unique_entity IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
                session.run("CREATE CONSTRAINT unique_document IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
                
                # Advanced vector indexes
                try:
                    session.run("""
                    CREATE VECTOR INDEX advanced_concept_embeddings IF NOT EXISTS
                    FOR (c:Concept) ON (c.voyage_embedding)
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 1024,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                    """)
                    
                    session.run("""
                    CREATE VECTOR INDEX document_voyage_embeddings IF NOT EXISTS
                    FOR (d:Document) ON (d.voyage_embedding)
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 1024,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                    """)
                except:
                    pass
                
                # Relationship indexes for traversal optimization
                session.run("CREATE INDEX relationship_strength IF NOT EXISTS FOR ()-[r:RELATES]-() ON (r.strength)")
                session.run("CREATE INDEX concept_importance IF NOT EXISTS FOR (c:Concept) ON (c.importance)")
                
                logger.info("✅ Advanced GraphRAG schema created")
                self.knowledge_graph_created = True
                self.vector_index_created = True
                
        except Exception as e:
            logger.error(f"⚠️ Failed to create advanced schema: {str(e)}")

    def advanced_entity_extraction(self, text: str) -> Dict:
        """Enhanced entity extraction with Voyage AI semantic analysis"""
        prompt = f"""Analyze this technical text and extract comprehensive knowledge structures.
Focus on technical specifications, relationships, and semantic connections.

Text: {text[:3000]}

Return a detailed JSON object with this structure:
{{
    "concepts": [
        {{
            "name": "concept_name",
            "description": "detailed technical description",
            "importance": 0.85,
            "type": "technical|specification|product|protocol",
            "semantic_tags": ["tag1", "tag2"],
            "technical_details": {{"param1": "value1"}}
        }}
    ],
    "entities": [
        {{
            "name": "entity_name", 
            "description": "detailed description",
            "type": "product|component|specification|value",
            "attributes": {{"attr1": "val1"}}
        }}
    ],
    "relationships": [
        {{
            "source": "source_name",
            "target": "target_name", 
            "type": "has_specification|operates_at|connects_to|relates_to",
            "strength": 0.8,
            "context": "relationship context"
        }}
    ],
    "technical_specifications": [
        {{
            "parameter": "voltage",
            "value": "3.3V",
            "unit": "volts",
            "context": "operating voltage"
        }}
    ]
}}"""
        
        try:
            response = self.llm.enhanced_chat_generate([
                {"role": "system", "content": "You are an expert technical knowledge extractor."},
                {"role": "user", "content": prompt}
            ], temperature=0.1)[0]
            
            # Extract and parse JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
                
                # Ensure all required fields
                parsed_data.setdefault('concepts', [])
                parsed_data.setdefault('entities', [])
                parsed_data.setdefault('relationships', [])
                parsed_data.setdefault('technical_specifications', [])
                
                return parsed_data
            else:
                return {"concepts": [], "entities": [], "relationships": [], "technical_specifications": []}
                
        except Exception as e:
            logger.error(f"Advanced entity extraction failed: {str(e)}")
            return {"concepts": [], "entities": [], "relationships": [], "technical_specifications": []}

    def store_advanced_knowledge_graph(self, knowledge: Dict, source_text: str):
        """Store knowledge with advanced Voyage AI embeddings"""
        if not self.neo4j_driver:
            return
            
        try:
            with self.neo4j_driver.session() as session:
                # Store concepts with Voyage embeddings
                for concept in knowledge.get('concepts', []):
                    try:
                        concept_text = f"{concept['name']} {concept.get('description', '')}"
                        
                        # Generate Voyage AI embedding
                        if concept_text not in self.voyage_embeddings_cache:
                            voyage_embedding = client.multimodal_embed(
                                inputs=[[concept_text]],
                                model="voyage-multimodal-3",
                                input_type="document"
                            ).embeddings[0]
                            self.voyage_embeddings_cache[concept_text] = voyage_embedding
                        else:
                            voyage_embedding = self.voyage_embeddings_cache[concept_text]
                        
                        session.run("""
                        MERGE (c:Concept {name: $name})
                        SET c.description = $description,
                            c.importance = $importance,
                            c.type = $type,
                            c.voyage_embedding = $voyage_embedding,
                            c.semantic_tags = $semantic_tags,
                            c.technical_details = $technical_details,
                            c.last_mentioned = datetime(),
                            c.mention_count = coalesce(c.mention_count, 0) + 1,
                            c.voyage_processed = true
                        """, {
                            **concept,
                            'voyage_embedding': voyage_embedding,
                            'semantic_tags': concept.get('semantic_tags', []),
                            'technical_details': json.dumps(concept.get('technical_details', {}))
                        })
                        
                    except Exception as e:
                        logger.error(f"Failed to store concept {concept.get('name')}: {str(e)}")
                
                # Store enhanced entities
                for entity in knowledge.get('entities', []):
                    session.run("""
                    MERGE (e:Entity {name: $name})
                    SET e.description = $description,
                        e.type = $type,
                        e.attributes = $attributes,
                        e.last_mentioned = datetime(),
                        e.mention_count = coalesce(e.mention_count, 0) + 1
                    """, {
                        **entity,
                        'attributes': json.dumps(entity.get('attributes', {}))
                    })
                
                # Store enhanced relationships
                for rel in knowledge.get('relationships', []):
                    session.run("""
                    MATCH (source) WHERE source.name = $source
                    MATCH (target) WHERE target.name = $target
                    MERGE (source)-[r:RELATES {type: $type}]->(target)
                    SET r.strength = $strength,
                        r.context = $context,
                        r.last_used = datetime(),
                        r.usage_count = coalesce(r.usage_count, 0) + 1,
                        r.voyage_enhanced = true
                    """, rel)
                
                # Store technical specifications as separate nodes
                for spec in knowledge.get('technical_specifications', []):
                    session.run("""
                    MERGE (s:Specification {parameter: $parameter})
                    SET s.value = $value,
                        s.unit = $unit,
                        s.context = $context,
                        s.last_updated = datetime()
                    """, spec)
                
                logger.info(f"✅ Advanced knowledge stored: {len(knowledge.get('concepts', []))} concepts with Voyage embeddings")
                
        except Exception as e:
            logger.error(f"⚠️ Advanced knowledge storage failed: {str(e)}")

    def advanced_graph_traversal(self, query: str, max_depth: int = 3) -> List[Dict]:
        """Advanced graph traversal with Voyage AI semantic routing"""
        if not self.neo4j_driver:
            return []
            
        try:
            # Generate Voyage embedding for query
            query_embedding = client.multimodal_embed(
                inputs=[[query]],
                model="voyage-multimodal-3",
                input_type="document"
            ).embeddings[0]
            
            with self.neo4j_driver.session() as session:
                # Multi-hop graph traversal with semantic similarity
                results = session.run("""
                CALL db.index.vector.queryNodes('advanced_concept_embeddings', 10, $embedding)
                YIELD node as start_node, score as start_score
                
                // Traverse relationships up to max_depth
                MATCH path = (start_node)-[:RELATES*1..3]-(connected)
                WHERE connected.voyage_processed = true
                
                WITH start_node, start_score, path, connected,
                     reduce(strength = 1.0, rel in relationships(path) | strength * rel.strength) as path_strength
                
                RETURN DISTINCT
                    start_node.name as primary_concept,
                    start_node.description as primary_description, 
                    start_score,
                    connected.name as connected_concept,
                    connected.description as connected_description,
                    connected.importance as connected_importance,
                    path_strength,
                    length(path) as path_length,
                    [rel in relationships(path) | rel.type] as relationship_types
                
                ORDER BY start_score DESC, path_strength DESC, connected_importance DESC
                LIMIT 15
                """, {"embedding": query_embedding, "max_depth": max_depth}).data()
                
                # Group and rank results
                graph_context = []
                for result in results:
                    context = {
                        "concept": result["primary_concept"],
                        "description": result["primary_description"],  
                        "score": float(result["start_score"]),
                        "connected_concepts": [{
                            "name": result["connected_concept"],
                            "description": result["connected_description"],
                            "importance": float(result.get("connected_importance", 0.5)),
                            "path_strength": float(result["path_strength"]),
                            "path_length": result["path_length"],
                            "relationship_types": result["relationship_types"]
                        }] if result["connected_concept"] else [],
                        "voyage_enhanced": True
                    }
                    graph_context.append(context)
                
                # Store traversal history for analytics
                self.graph_traversal_history[query].append({
                    "timestamp": datetime.now().isoformat(),
                    "results_count": len(graph_context),
                    "max_score": max([ctx["score"] for ctx in graph_context]) if graph_context else 0.0
                })
                
                return graph_context
                
        except Exception as e:
            logger.error(f"⚠️ Advanced graph traversal failed: {str(e)}")
            return []

    def hybrid_voyage_search(self, query: str, top_k: int = 5) -> Dict:
        """Advanced hybrid search combining Voyage AI embeddings with GraphRAG"""
        start_time = datetime.now()
        
        # Multi-strategy search approach
        search_results = {
            "voyage_vector_results": [],
            "graph_traversal_results": [],
            "faiss_semantic_results": [],
            "keyword_matches": [],
            "combined_results": []
        }
        
        try:
            # 1. Voyage AI vector search
            query_embedding = client.multimodal_embed(
                inputs=[[query]],
                model="voyage-multimodal-3",
                input_type="document"
            ).embeddings[0]
            
            if self.neo4j_driver:
                with self.neo4j_driver.session() as session:
                    # Search concepts with Voyage embeddings
                    vector_results = session.run("""
                    CALL db.index.vector.queryNodes('advanced_concept_embeddings', $top_k, $embedding)
                    YIELD node, score
                    RETURN 
                        node.name as concept,
                        node.description as description,
                        score,
                        node.importance as importance,
                        node.type as type,
                        node.semantic_tags as tags
                    ORDER BY score DESC
                    """, {"embedding": query_embedding, "top_k": top_k}).data()
                    
                    search_results["voyage_vector_results"] = [
                        {
                            "concept": r["concept"],
                            "description": r["description"],
                            "score": float(r["score"]),
                            "type": "voyage_vector",
                            "metadata": {
                                "importance": r.get("importance", 0.5),
                                "type": r.get("type", "unknown"),
                                "tags": r.get("tags", [])
                            }
                        } for r in vector_results
                    ]
            
            # 2. Graph traversal search
            graph_results = self.advanced_graph_traversal(query)
            search_results["graph_traversal_results"] = [
                {
                    "concept": r["concept"],
                    "description": r["description"],
                    "score": r["score"],
                    "type": "graph_traversal",
                    "metadata": {
                        "connected_concepts": r["connected_concepts"],
                        "voyage_enhanced": r.get("voyage_enhanced", False)
                    }
                } for r in graph_results[:top_k]
            ]
            
            # 3. Combine and rank results (advanced fusion algorithm)
            combined = []
            
            # Add vector results with high weight
            for vr in search_results["voyage_vector_results"]:
                combined.append({
                    **vr,
                    "combined_score": vr["score"] * 1.2  # Higher weight for direct matches
                })
            
            # Add graph results with context weighting
            for gr in search_results["graph_traversal_results"]:
                context_boost = 1.0 + (0.1 * len(gr["metadata"]["connected_concepts"]))
                combined.append({
                    **gr,
                    "combined_score": gr["score"] * 0.9 * context_boost  # Slightly lower base weight but context boosted
                })
            
            # Sort by combined score and deduplicate
            seen_concepts = set()
            final_results = []
            for item in sorted(combined, key=lambda x: x["combined_score"], reverse=True):
                if item["concept"] not in seen_concepts:
                    final_results.append(item)
                    seen_concepts.add(item["concept"])
                    if len(final_results) >= top_k:
                        break
            
            search_results["combined_results"] = final_results[:top_k]
            search_results["processing_time"] = (datetime.now() - start_time).total_seconds()
            
            return search_results
            
        except Exception as e:
            logger.error(f"⚠️ Hybrid search failed: {str(e)}")
            return {
                "error": str(e),
                "combined_results": []
            }

    def process_pdf_document(self, file_path: str):
        """Advanced PDF processing with multimodal capabilities"""
        try:
            doc = fitz.open(file_path)
            full_text = ""
            images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                full_text += page.get_text()
                
                # Extract images for potential multimodal processing
                for img in page.get_images():
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    images.append(image)
            
            # Enhanced knowledge extraction
            knowledge = self.advanced_entity_extraction(full_text)
            self.store_advanced_knowledge_graph(knowledge, full_text)
            
            # Process images if needed (placeholder for multimodal)
            if images:
                logger.info(f"Extracted {len(images)} images (multimodal processing not implemented)")
            
            return {
                "status": "success",
                "pages": len(doc),
                "characters": len(full_text),
                "concepts_extracted": len(knowledge.get("concepts", [])),
                "entities_extracted": len(knowledge.get("entities", [])),
                "relationships_found": len(knowledge.get("relationships", []))
            }
            
        except Exception as e:
            logger.error(f"⚠️ PDF processing failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def generate_response(self, query: str, chat_history: List[Dict] = None) -> Dict:
        """Enhanced response generation with GraphRAG context"""
        start_time = datetime.now()
        chat_history = chat_history or []
        
        try:
            # 1. Perform hybrid search
            search_results = self.hybrid_voyage_search(query)
            graph_context = search_results.get("combined_results", [])
            
            # 2. Prepare LLM prompt with GraphRAG context
            messages = [
                {
                    "role": "system",
                    "content": """You are a technical expert assistant with access to a knowledge graph.
Use the provided GraphRAG context to give accurate, detailed responses. Cite concepts when appropriate."""
                }
            ]
            
            # Add chat history
            for msg in chat_history[-6:]:  # Last 3 exchanges
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add current query
            messages.append({"role": "user", "content": query})
            
            # 3. Generate response with confidence scoring
            response, confidence, processing_time = self.llm.enhanced_chat_generate(
                messages,
                graph_context=graph_context,
                temperature=0.3
            )
            
            # 4. Prepare analytics
            analytics = {
                "search_strategy": "hybrid_voyage",
                "context_used": len(graph_context),
                "graph_context_used": len([c for c in graph_context if c["type"] == "graph_traversal"]),
                "voyage_embedding_used": True,
                "relevance_scores": {c["concept"]: c["score"] for c in graph_context},
                "extracted_entities": self.advanced_entity_extraction(query).get("entities", []),
                "graph_paths": [c["metadata"].get("connected_concepts", []) for c in graph_context],
                "response_confidence": confidence,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "additional_metadata": {
                    "voyage_version": "multimodal-3",
                    "graph_depth": 3,
                    "llm_model": self.llm.model
                }
            }
            
            return {
                "response": response,
                "context": graph_context,
                "analytics": analytics,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"⚠️ Response generation failed: {str(e)}")
            return {
                "response": f"Error: {str(e)}",
                "context": [],
                "analytics": {},
                "status": "error"
            }