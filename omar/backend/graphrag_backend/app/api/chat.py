from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
    chat_history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    response: str
    chat_id: str
    context: List[Dict]
    analytics: Dict
    status: str

@router.post("/send", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    try:
        from app.main import graph_rag, db_manager
        
        # Create new chat if needed
        chat_id = request.chat_id
        if not chat_id:
            chat_id = db_manager.create_chat(f"Mobile Chat {uuid.uuid4()}")
        
        # Add user message to DB
        db_manager.add_enhanced_message(chat_id, "user", request.message)
        
        # Generate response
        history = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
        response = graph_rag.generate_response(request.message, history)
        
        # Add assistant message to DB
        if response.get("analytics"):
            db_manager.add_enhanced_message(
                chat_id, 
                "assistant", 
                response["response"], 
                response["analytics"]
            )
        
        return ChatResponse(
            response=response["response"],
            chat_id=chat_id,
            context=response.get("context", []),
            analytics=response.get("analytics", {}),
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{chat_id}")
async def get_chat_history(chat_id: str):
    try:
        from app.main import db_manager
        messages = db_manager.get_chat_messages(chat_id)
        return {"messages": messages, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_chats():
    try:
        from app.main import db_manager
        chats = db_manager.get_chats()
        return {"chats": chats, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{chat_id}")
async def delete_chat(chat_id: str):
    try:
        from app.main import db_manager
        success = db_manager.delete_chat(chat_id)
        return {"success": success, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))