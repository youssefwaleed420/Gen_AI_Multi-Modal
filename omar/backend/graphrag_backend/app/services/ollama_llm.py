import requests
import json
from typing import List, Dict, Tuple
from datetime import datetime
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:1b"  # or whatever model you've pulled locally

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)




class AdvancedOllamaLLM:
    """Enhanced Ollama LLM with GraphRAG integration"""
    
    def __init__(self, base_url: str = OLLAMA_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url
        self.chat_url = OLLAMA_CHAT_URL
        self.model = model
        self.check_ollama_connection()
    
    def check_ollama_connection(self):
        """Enhanced connection check with model verification"""
        try:
            response = requests.get(f"{self.base_url.replace('/api/generate', '')}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.model not in model_names:
                    logger.warning(f"⚠️ Model '{self.model}' not found. Available: {model_names}")
                    if model_names:
                        self.model = model_names[0]
                        logger.info(f"🔄 Using available model: {self.model}")
                else:
                    logger.info(f"✅ Ollama connected with model: {self.model}")
            else:
                logger.error("❌ Ollama server not responding")
        except Exception as e:
            logger.error(f"❌ Ollama connection failed: {str(e)}")
    
    def enhanced_chat_generate(self, messages: List[Dict], graph_context: List[Dict] = None, **kwargs) -> Tuple[str, float]:
        """Enhanced chat generation with confidence scoring"""
        try:
            # Add graph context to system message if available
            if graph_context and messages:
                graph_info = self._format_graph_context(graph_context)
                if messages[0]["role"] == "system":
                    messages[0]["content"] += f"\n\n### Enhanced Knowledge Graph Context:\n{graph_info}"
                else:
                    messages.insert(0, {"role": "system", "content": f"### Knowledge Graph Context:\n{graph_info}"})
            
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                    "num_ctx": kwargs.get("num_ctx", 4096),
                    "repeat_penalty": kwargs.get("repeat_penalty", 1.1)
                }
            }
            
            start_time = datetime.now()
            response = requests.post(self.chat_url, json=payload, timeout=120)
            response.raise_for_status()
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = response.json()
            content = result.get("message", {}).get("content", "No response from LLM")
            
            # Calculate confidence based on response quality
            confidence = self._calculate_response_confidence(content, graph_context)
            
            return content, confidence, processing_time
            
        except Exception as e:
            logger.error(f"⚠️ Enhanced chat generation failed: {str(e)}")
            return f"Error: {str(e)}", 0.0, 0.0
    
    def _format_graph_context(self, graph_context: List[Dict]) -> str:
        """Format graph context for LLM consumption"""
        formatted = []
        for ctx in graph_context[:5]:  # Top 5 contexts
            concept = ctx.get('concept', 'Unknown')
            description = ctx.get('description', 'No description')
            score = ctx.get('score', 0.0)
            related = ctx.get('related_items', [])
            
            formatted.append(f"**{concept}** (relevance: {score:.3f})")
            formatted.append(f"  Description: {description}")
            if related:
                related_names = [item.get('name', '') for item in related[:3]]
                formatted.append(f"  Related: {', '.join(related_names)}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _calculate_response_confidence(self, response: str, graph_context: List[Dict] = None) -> float:
        """Calculate response confidence based on multiple factors"""
        confidence = 0.5  # Base confidence
        
        # Length factor (longer responses often more comprehensive)
        length_factor = min(len(response) / 500, 1.0) * 0.2
        confidence += length_factor
        
        # Graph context utilization
        if graph_context:
            context_concepts = [ctx.get('concept', '').lower() for ctx in graph_context]
            response_lower = response.lower()
            concept_mentions = sum(1 for concept in context_concepts if concept in response_lower)
            context_factor = min(concept_mentions / len(context_concepts), 1.0) * 0.3
            confidence += context_factor
        
        # Technical terms and specificity
        technical_indicators = ['specification', 'parameter', 'voltage', 'frequency', 'module', 'protocol']
        tech_mentions = sum(1 for term in technical_indicators if term in response.lower())
        tech_factor = min(tech_mentions / 3, 1.0) * 0.2
        confidence += tech_factor
        
        return min(confidence, 1.0)
