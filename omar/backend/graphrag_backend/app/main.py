# File: graphrag_backend/app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn
import sys
import os
from pathlib import Path

# Add the project root to the Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir.parent))
sys.path.insert(0, str(current_dir))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import modules with multiple fallback paths
db_manager_class = None
analytics_router = None

# Try different import paths for database manager
for import_path in [
    "graphrag_backend.app.services.database_manager",
    "app.services.database_manager", 
    "services.database_manager",
    "database_manager"
]:
    try:
        module = __import__(import_path, fromlist=['DatabaseManager'])
        db_manager_class = getattr(module, 'DatabaseManager')
        logger.info(f"✅ Successfully imported DatabaseManager from {import_path}")
        break
    except (ImportError, AttributeError) as e:
        logger.debug(f"Failed to import from {import_path}: {e}")
        continue

# If no database manager found, create a minimal one
if not db_manager_class:
    logger.warning("⚠️ Could not import DatabaseManager, creating minimal version")
    
    import psycopg2
    import psycopg2.extras
    import uuid
    import json
    from typing import List, Dict, Optional
    from datetime import datetime
    
    # Use your database config
    DB_CONFIG = {
        "host": "aws-0-us-west-1.pooler.supabase.com",
        "port": 6543,
        "database": "postgres",
        "user": "postgres.jsgdmwbhzvwdhardoimb",
        "password": "omernasser123"
    }
    
    class MinimalDatabaseManager:
        """Minimal database manager for basic functionality"""
        
        def __init__(self):
            self.connection = None
            self.connect()
        
        def connect(self):
            """Connect to PostgreSQL database"""
            try:
                self.connection = psycopg2.connect(**DB_CONFIG)
                logger.info("✅ Successfully connected to PostgreSQL!")
            except Exception as e:
                logger.error(f"❌ Failed to connect to database: {str(e)}")
                self.connection = None
        
        def get_system_health(self) -> Dict:
            """Get basic system health"""
            if not self.connection:
                return {"status": "unhealthy", "error": "No database connection"}
            
            try:
                cursor = self.connection.cursor()
                cursor.execute("SELECT 1 as test;")
                cursor.fetchone()
                cursor.close()
                
                return {
                    "status": "healthy",
                    "database_connected": True,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {
                    "status": "unhealthy", 
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        def close(self):
            """Close database connection"""
            if self.connection:
                self.connection.close()
                logger.info("🔚 Database connection closed.")
    
    db_manager_class = MinimalDatabaseManager

# Try different import paths for analytics router
for import_path in [
    "graphrag_backend.app.api.analytics",
    "app.api.analytics",
    "api.analytics",
    "analytics"
]:
    try:
        module = __import__(import_path, fromlist=['router'])
        analytics_router = getattr(module, 'router')
        logger.info(f"✅ Successfully imported analytics router from {import_path}")
        break
    except (ImportError, AttributeError) as e:
        logger.debug(f"Failed to import analytics router from {import_path}: {e}")
        continue

# If no analytics router found, create a minimal one
if not analytics_router:
    logger.warning("⚠️ Could not import analytics router, creating minimal version")
    
    from fastapi import APIRouter
    
    analytics_router = APIRouter(tags=["analytics"])
    
    @analytics_router.get("/health")
    async def minimal_health():
        """Minimal health check"""
        return {"status": "ok", "message": "Analytics router is working"}

# Global database manager instance
db_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global db_manager
    
    # Startup
    logger.info("🚀 Starting GraphRAG Backend API...")
    try:
        if db_manager_class:
            db_manager = db_manager_class()
            if hasattr(db_manager, 'connection') and db_manager.connection:
                logger.info("✅ Database connection established")
            else:
                logger.error("❌ Failed to establish database connection")
        else:
            logger.error("❌ No database manager class available")
    except Exception as e:
        logger.error(f"❌ Startup error: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down GraphRAG Backend API...")
    try:
        if db_manager and hasattr(db_manager, 'close'):
            db_manager.close()
            logger.info("✅ Database connection closed")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {str(e)}")

# Create FastAPI app with lifespan events
app = FastAPI(
    title="GraphRAG Backend API",
    description="Enhanced GraphRAG backend with analytics and chat management",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include analytics router if available
if analytics_router:
    app.include_router(analytics_router, prefix="/api/v1/analytics")
    logger.info("✅ Analytics router included")
else:
    logger.warning("⚠️ No analytics router available")

# Basic endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "GraphRAG Backend API is running",
        "status": "healthy",
        "version": "1.0.0",
        "features": {
            "database_manager": db_manager_class is not None,
            "analytics_router": analytics_router is not None
        }
    }

@app.get("/health")
async def health_check():
    """Basic health check"""
    try:
        health_data = {
            "status": "healthy",
            "message": "Service is running",
            "components": {
                "database_manager": "available" if db_manager_class else "unavailable",
                "analytics_router": "available" if analytics_router else "unavailable"
            }
        }
        
        # Test database connection if available
        if db_manager and hasattr(db_manager, 'get_system_health'):
            db_health = db_manager.get_system_health()
            health_data["database"] = db_health
        elif db_manager and hasattr(db_manager, 'connection') and db_manager.connection:
            try:
                cursor = db_manager.connection.cursor()
                cursor.execute("SELECT 1;")
                cursor.fetchone()
                cursor.close()
                health_data["database"] = {"status": "connected"}
            except Exception as e:
                health_data["database"] = {"status": "error", "error": str(e)}
        else:
            health_data["database"] = {"status": "unavailable"}
        
        return health_data
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# Chat management endpoints (only if database manager is available)
if db_manager_class:
    @app.post("/api/v1/chats")
    async def create_chat(title: str, user_id: str = "default_user"):
        """Create a new chat"""
        try:
            if not db_manager or not hasattr(db_manager, 'connection') or not db_manager.connection:
                raise HTTPException(status_code=503, detail="Database not available")
            
            if hasattr(db_manager, 'create_chat'):
                chat_id = db_manager.create_chat(title, user_id)
                if chat_id:
                    return {
                        "message": "Chat created successfully",
                        "chat_id": chat_id,
                        "title": title,
                        "user_id": user_id
                    }
                else:
                    raise HTTPException(status_code=500, detail="Failed to create chat")
            else:
                raise HTTPException(status_code=501, detail="Create chat not implemented")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to create chat: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/chats/{chat_id}/messages")
    async def add_message(
        chat_id: str, 
        role: str, 
        content: str, 
        search_metadata: dict = None
    ):
        """Add a message to a chat"""
        try:
            if not db_manager or not hasattr(db_manager, 'connection') or not db_manager.connection:
                raise HTTPException(status_code=503, detail="Database not available")
            
            if role not in ["user", "assistant", "system"]:
                raise HTTPException(status_code=400, detail="Invalid role")
            
            if hasattr(db_manager, 'add_enhanced_message'):
                success = db_manager.add_enhanced_message(chat_id, role, content, search_metadata)
            elif hasattr(db_manager, 'add_message'):
                success = db_manager.add_message(chat_id, role, content)
            else:
                raise HTTPException(status_code=501, detail="Add message not implemented")
            
            if success:
                return {
                    "message": "Message added successfully",
                    "chat_id": chat_id,
                    "role": role
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to add message")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to add message: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Debug endpoint to show current module structure
@app.get("/debug/modules")
async def debug_modules():
    """Debug endpoint to show what modules are available"""
    return {
        "python_path": sys.path[:5],  # Show first 5 entries
        "current_directory": str(Path.cwd()),
        "main_file_location": str(Path(__file__)),
        "available_modules": {
            "database_manager": db_manager_class.__name__ if db_manager_class else "Not available",
            "analytics_router": "Available" if analytics_router else "Not available"
        },
        "environment": {
            "PYTHONPATH": os.environ.get("PYTHONPATH", "Not set"),
        }
    }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return HTTPException(
        status_code=500,
        detail="An internal server error occurred"
    )

if __name__ == "__main__":
    # For development only
    print("🚀 Starting GraphRAG Backend API...")
    print(f"📁 Current directory: {Path.cwd()}")
    print(f"📄 Main file: {Path(__file__)}")
    print(f"🐍 Python path: {sys.path[:3]}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )