# File: graphrag_backend/app/api/analytics.py

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
from graphrag_backend.app.db.database import DatabaseManager

# You'll need to import your database manager here
from graphrag_backend.app.services.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analytics"])

# Dependency to get database manager instance
def get_database_manager():
    """Dependency to get database manager instance"""
    try:
        # Import here to avoid circular imports
        from graphrag_backend.app.services.database_manager import DatabaseManager
        return DatabaseManager()
    except Exception as e:
        logger.error(f"Failed to get database manager: {str(e)}")
        raise HTTPException(status_code=500, detail="Database connection failed")

@router.get("/health")
async def get_system_health(db: DatabaseManager = Depends(get_database_manager)):
    """Get system health metrics"""
    try:
        health_data = db.get_system_health()
        return health_data
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@router.get("/user/{user_id}/stats")
async def get_user_statistics(
    user_id: str,
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get comprehensive user statistics"""
    try:
        stats = db.get_user_stats(user_id)
        return {
            "user_id": user_id,
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get user stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@router.get("/chat/{chat_id}/analytics")
async def get_chat_analytics(
    chat_id: str,
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get comprehensive chat analytics"""
    try:
        analytics = db.get_chat_analytics(chat_id)
        return {
            "chat_id": chat_id,
            "analytics": analytics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get chat analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@router.get("/chats")
async def get_all_chats(
    user_id: str = "default_user",
    limit: int = 50,
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get all chats for a user with analytics"""
    try:
        chats = db.get_chats(user_id, limit)
        return {
            "user_id": user_id,
            "chats": chats,
            "total_returned": len(chats),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get chats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@router.get("/chat/{chat_id}/messages")
async def get_chat_messages(
    chat_id: str,
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get all messages for a specific chat"""
    try:
        messages = db.get_chat_messages(chat_id)
        return {
            "chat_id": chat_id,
            "messages": messages,
            "total_messages": len(messages),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get chat messages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@router.post("/cleanup")
async def cleanup_old_sessions(
    days_old: int = 30,
    db: DatabaseManager = Depends(get_database_manager)
):
    """Clean up old inactive sessions"""
    try:
        if days_old < 1:
            raise HTTPException(status_code=400, detail="days_old must be at least 1")
        
        deleted_count = db.cleanup_old_sessions(days_old)
        return {
            "message": f"Cleaned up {deleted_count} old sessions",
            "days_old": days_old,
            "deleted_count": deleted_count,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cleanup sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@router.delete("/chat/{chat_id}")
async def delete_chat(
    chat_id: str,
    db: DatabaseManager = Depends(get_database_manager)
):
    """Delete a specific chat"""
    try:
        success = db.delete_chat(chat_id)
        if success:
            return {
                "message": f"Chat {chat_id} deleted successfully",
                "chat_id": chat_id,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Chat not found or could not be deleted")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@router.post("/knowledge-analytics")
async def add_knowledge_analytics(
    session_id: str,
    query_text: str,
    extracted_concepts: Optional[List[Dict]] = None,
    graph_traversal_path: Optional[List[Dict]] = None,
    semantic_scores: Optional[Dict] = None,
    voyage_embedding: Optional[List[float]] = None,
    db: DatabaseManager = Depends(get_database_manager)
):
    """Add knowledge analytics data"""
    try:
        success = db.add_knowledge_analytics(
            session_id=session_id,
            query_text=query_text,
            extracted_concepts=extracted_concepts,
            graph_traversal_path=graph_traversal_path,
            semantic_scores=semantic_scores,
            voyage_embedding=voyage_embedding
        )
        
        if success:
            return {
                "message": "Knowledge analytics added successfully",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add knowledge analytics")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add knowledge analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@router.get("/dashboard")
async def get_dashboard_data(
    user_id: str = "default_user",
    db: DatabaseManager = Depends(get_database_manager)
):
    """Get comprehensive dashboard data"""
    try:
        # Get user stats
        user_stats = db.get_user_stats(user_id)
        
        # Get recent chats
        recent_chats = db.get_chats(user_id, limit=10)
        
        # Get system health
        system_health = db.get_system_health()
        
        return {
            "user_id": user_id,
            "user_statistics": user_stats,
            "recent_chats": recent_chats,
            "system_health": system_health,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()