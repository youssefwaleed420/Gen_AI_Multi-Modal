from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile
import os
from fastapi import APIRouter


router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        from app.main import graph_rag
        
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files supported")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Process with GraphRAG
        result = graph_rag.process_pdf_document(tmp_path)
        
        # Cleanup
        os.unlink(tmp_path)
        
        return {
            "filename": file.filename,
            "processing_result": result,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graph-stats")
async def get_graph_statistics():
    try:
        from app.main import graph_rag
        if not graph_rag.neo4j_driver:
            raise HTTPException(status_code=503, detail="Neo4j not connected")
        
        # Add your graph statistics logic here
        return {"stats": "placeholder", "status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))