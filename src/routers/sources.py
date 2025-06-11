# /home/ubuntu/open_notebook_full_backend/fastapi_backend/src/routers/sources.py

from fastapi import (
    APIRouter, Depends, HTTPException, status, UploadFile, File, Form
)
from typing import List, Optional, Dict, Any
from datetime import datetime

from surrealdb import AsyncSurreal

from ..database import get_db_connection
from ..models import (
    Source, SourceSummary, StatusResponse, TaskStatus, SourceResponse
)

# Create a router for source-related endpoints
router = APIRouter(
    tags=["Sources"],
)

# Define the table name for sources in SurrealDB
SOURCE_TABLE = "source"
NOTEBOOK_TABLE = "notebook" # Needed for context checks

# --- Placeholder Models for Source Creation ---
# These should be moved to models.py eventually
from pydantic import BaseModel, HttpUrl

class SourceURLCreate(BaseModel):
    url: HttpUrl

class SourceTextCreate(BaseModel):
    title: str
    content: str

class SourceYouTubeCreate(BaseModel):
    url: HttpUrl

# --- Endpoints ---

@router.post("/api/v1/notebooks/by-name/{name}/sources/url", response_model=TaskStatus, status_code=status.HTTP_202_ACCEPTED)
async def add_source_from_url_by_name(
    name: str,
    source_data: SourceURLCreate,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Adds a new source by scraping a URL to a notebook specified by name."""
    try:
        # First get the notebook by name
        query = f"SELECT * FROM {NOTEBOOK_TABLE} WHERE name = $name"
        bindings = {"name": name}
        result = await db.query(query, bindings)
        
        if not result or len(result) == 0:
            raise HTTPException(status_code=404, detail=f"Notebook with name '{name}' not found")
            
        notebook = result[0]
        notebook_id = notebook['id']
        if hasattr(notebook_id, 'table_name') and hasattr(notebook_id, 'record_id'):
            notebook_id = f"{notebook_id.table_name}:{notebook_id.record_id}"
        else:
            notebook_id = str(notebook_id)

        # Placeholder implementation:
        print(f"Received request to add URL {source_data.url} to notebook {name}")
        # In a real app, create DB record and trigger background task here
        task_id = f"task_url_{datetime.utcnow().timestamp()}" # Example task ID
        return TaskStatus(task_id=task_id, status="pending", message="URL processing started.")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error adding URL source to notebook {name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error adding URL source: {e}")

@router.post("/api/v1/notebooks/{notebook_id}/sources/url", response_model=TaskStatus, status_code=status.HTTP_202_ACCEPTED)
async def add_source_from_url(
    notebook_id: str,
    source_data: SourceURLCreate,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Adds a new source by scraping a URL. Triggers background processing."""
    if ":" not in notebook_id:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid notebook ID format.")
    # 1. Verify notebook exists
    # 2. Create a placeholder source record in DB with status 'pending'
    # 3. Trigger a background task (e.g., Celery, ARQ) to scrape and process the URL
    # 4. Return the task ID
    # Placeholder implementation:
    print(f"Received request to add URL {source_data.url} to notebook {notebook_id}")
    # In a real app, create DB record and trigger background task here
    task_id = f"task_url_{datetime.utcnow().timestamp()}" # Example task ID
    return TaskStatus(task_id=task_id, status="pending", message="URL processing started.")

@router.post("/api/v1/notebooks/by-name/{name}/sources/upload", response_model=TaskStatus, status_code=status.HTTP_202_ACCEPTED)
async def add_source_from_upload_by_name(
    name: str,
    file: UploadFile = File(...),
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Adds a new source from an uploaded file to a notebook specified by name."""
    try:
        # First get the notebook by name
        query = f"SELECT * FROM {NOTEBOOK_TABLE} WHERE name = $name"
        bindings = {"name": name}
        result = await db.query(query, bindings)
        
        if not result or len(result) == 0:
            raise HTTPException(status_code=404, detail=f"Notebook with name '{name}' not found")
            
        notebook = result[0]
        notebook_id = notebook['id']
        if hasattr(notebook_id, 'table_name') and hasattr(notebook_id, 'record_id'):
            notebook_id = f"{notebook_id.table_name}:{notebook_id.record_id}"
        else:
            notebook_id = str(notebook_id)

        # Placeholder implementation:
        print(f"Received file upload {file.filename} ({file.content_type}) for notebook {name}")
        # In a real app, save file, create DB record, trigger background task
        task_id = f"task_upload_{datetime.utcnow().timestamp()}" # Example task ID
        return TaskStatus(task_id=task_id, status="pending", message=f"File '{file.filename}' processing started.")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error adding file source to notebook {name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error adding file source: {e}")

@router.post("/api/v1/notebooks/{notebook_id}/sources/upload", response_model=TaskStatus, status_code=status.HTTP_202_ACCEPTED)
async def add_source_from_upload(
    notebook_id: str,
    file: UploadFile = File(...),
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Adds a new source from an uploaded file (PDF, DOCX, etc.). Triggers background processing."""
    if ":" not in notebook_id:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid notebook ID format.")
    # 1. Verify notebook exists
    # 2. Save the uploaded file temporarily
    # 3. Create a placeholder source record in DB with status 'pending', store filename/type
    # 4. Trigger a background task to extract text and process the file
    # 5. Return the task ID
    # Placeholder implementation:
    print(f"Received file upload {file.filename} ({file.content_type}) for notebook {notebook_id}")
    # In a real app, save file, create DB record, trigger background task
    task_id = f"task_upload_{datetime.utcnow().timestamp()}" # Example task ID
    return TaskStatus(task_id=task_id, status="pending", message=f"File '{file.filename}' processing started.")

@router.post("/api/v1/notebooks/by-name/{name}/sources/text", response_model=SourceResponse, status_code=status.HTTP_201_CREATED)
async def add_source_from_text_by_name(
    name: str,
    source_data: SourceTextCreate,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Adds a new text source to a notebook specified by name."""
    try:
        # Get notebook by name
        query = f"SELECT * FROM {NOTEBOOK_TABLE} WHERE name = $name"
        bindings = {"name": name}
        result = await db.query(query, bindings)
        
        if not result or len(result) == 0:
            raise HTTPException(status_code=404, detail=f"Notebook with name '{name}' not found")
            
        notebook = dict(result[0])
        notebook_id = notebook['id']
        if hasattr(notebook_id, 'table_name') and hasattr(notebook_id, 'record_id'):
            notebook_id = f"{notebook_id.table_name}:{notebook_id.record_id}"
        else:
            notebook_id = str(notebook_id)

        # Create source with all necessary fields
        data_to_create = {
            "title": source_data.title,
            "full_text": source_data.content,
            "type": "text",
            "status": "completed",
            "created": datetime.utcnow(),
            "updated": datetime.utcnow(),
            "notebook_id": notebook_id,
            "metadata": {"title": source_data.title},
            "insights": [],
            "embedded_chunks": 0
        }

        created_sources = await db.create(SOURCE_TABLE, data_to_create)
        
        if not created_sources:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create text source")
            
        source_dict = dict(created_sources[0] if isinstance(created_sources, list) else created_sources)
        source_dict = convert_record_id_to_string(source_dict)
        
        return SourceResponse(
            id=str(source_dict.get('id', '')),
            title=str(source_dict.get('title', '')),
            type=str(source_dict.get('type', '')),
            status=str(source_dict.get('status', '')),
            created=source_dict.get('created'),
            updated=source_dict.get('updated'),
            metadata=source_dict.get('metadata', {}),
            full_text=source_dict.get('full_text', ''),
            notebook_id=str(source_dict.get('notebook_id', '')),
            insights=source_dict.get('insights', []),
            embedded_chunks=source_dict.get('embedded_chunks', 0)
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error creating text source for notebook {name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error creating text source: {e}")

@router.post("/api/v1/notebooks/{notebook_id}/sources/text", response_model=SourceResponse, status_code=status.HTTP_201_CREATED)
async def add_source_from_text(
    notebook_id: str,
    source_data: SourceTextCreate,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Adds a new source directly from text content."""
    if ":" not in notebook_id:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid notebook ID format.")
    try:
        notebook = await db.select(notebook_id)
        if not notebook:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Notebook 	{notebook_id}	 not found")

        data_to_create = source_data.model_dump()
        data_to_create["type"] = "text"
        data_to_create["status"] = "completed" # Text sources are immediately complete
        data_to_create["created"] = datetime.utcnow()
        data_to_create["notebook_id"] = notebook_id
        data_to_create["metadata"] = {"title": source_data.title}

        created_sources = await db.create(SOURCE_TABLE, data_to_create)

        if created_sources and isinstance(created_sources, list) and len(created_sources) > 0:
            return created_sources[0]
        elif created_sources:
             return created_sources
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create text source in DB")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error creating text source for notebook {notebook_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error creating text source")

@router.post("/api/v1/notebooks/by-name/{name}/sources/youtube", response_model=TaskStatus, status_code=status.HTTP_202_ACCEPTED)
async def add_source_from_youtube_by_name(
    name: str,
    source_data: SourceYouTubeCreate,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Adds a new source from a YouTube URL to a notebook specified by name."""
    try:
        # First get the notebook by name
        query = f"SELECT * FROM {NOTEBOOK_TABLE} WHERE name = $name"
        bindings = {"name": name}
        result = await db.query(query, bindings)
        
        if not result or len(result) == 0:
            raise HTTPException(status_code=404, detail=f"Notebook with name '{name}' not found")
            
        notebook = result[0]
        notebook_id = notebook['id']
        if hasattr(notebook_id, 'table_name') and hasattr(notebook_id, 'record_id'):
            notebook_id = f"{notebook_id.table_name}:{notebook_id.record_id}"
        else:
            notebook_id = str(notebook_id)

        # Placeholder implementation:
        print(f"Received request to add YouTube URL {source_data.url} to notebook {name}")
        task_id = f"task_youtube_{datetime.utcnow().timestamp()}"
        return TaskStatus(task_id=task_id, status="pending", message="YouTube transcription started.")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error adding YouTube source to notebook {name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error adding YouTube source: {e}")

@router.post("/api/v1/notebooks/{notebook_id}/sources/youtube", response_model=TaskStatus, status_code=status.HTTP_202_ACCEPTED)
async def add_source_from_youtube(
    notebook_id: str,
    source_data: SourceYouTubeCreate,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Adds a new source from a YouTube URL. Triggers background processing for transcription."""
    if ":" not in notebook_id:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid notebook ID format.")
    # 1. Verify notebook exists
    # 2. Create placeholder source record
    # 3. Trigger background task for transcription
    # 4. Return task ID
    # Placeholder implementation:
    print(f"Received request to add YouTube URL {source_data.url} to notebook {notebook_id}")
    task_id = f"task_youtube_{datetime.utcnow().timestamp()}"
    return TaskStatus(task_id=task_id, status="pending", message="YouTube transcription started.")

@router.post("/api/v1/notebooks/by-name/{name}/sources/media", response_model=TaskStatus, status_code=status.HTTP_202_ACCEPTED)
async def add_source_from_media_by_name(
    name: str,
    file: UploadFile = File(...),
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Adds a new source from an uploaded audio/video file to a notebook specified by name."""
    try:
        # First get the notebook by name
        query = f"SELECT * FROM {NOTEBOOK_TABLE} WHERE name = $name"
        bindings = {"name": name}
        result = await db.query(query, bindings)
        
        if not result or len(result) == 0:
            raise HTTPException(status_code=404, detail=f"Notebook with name '{name}' not found")
            
        notebook = result[0]
        notebook_id = notebook['id']
        if hasattr(notebook_id, 'table_name') and hasattr(notebook_id, 'record_id'):
            notebook_id = f"{notebook_id.table_name}:{notebook_id.record_id}"
        else:
            notebook_id = str(notebook_id)

        # Placeholder implementation:
        print(f"Received media upload {file.filename} ({file.content_type}) for notebook {name}")
        task_id = f"task_media_{datetime.utcnow().timestamp()}"
        return TaskStatus(task_id=task_id, status="pending", message=f"Media file '{file.filename}' transcription started.")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error adding media source to notebook {name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error adding media source: {e}")

@router.post("/api/v1/notebooks/{notebook_id}/sources/media", response_model=TaskStatus, status_code=status.HTTP_202_ACCEPTED)
async def add_source_from_media(
    notebook_id: str,
    file: UploadFile = File(...),
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Adds a new source from an uploaded audio/video file. Triggers background processing for transcription."""
    if ":" not in notebook_id:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid notebook ID format.")
    # 1. Verify notebook exists
    # 2. Save file
    # 3. Create placeholder source record
    # 4. Trigger background task for transcription
    # 5. Return task ID
    # Placeholder implementation:
    print(f"Received media upload {file.filename} ({file.content_type}) for notebook {notebook_id}")
    task_id = f"task_media_{datetime.utcnow().timestamp()}"
    return TaskStatus(task_id=task_id, status="pending", message=f"Media file '{file.filename}' transcription started.")

@router.get("/api/v1/notebooks/by-name/{name}/sources", response_model=List[SourceSummary])
async def list_sources_for_notebook_by_name(
    name: str,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Lists all sources associated with a specific notebook by name."""
    try:
        # First get the notebook by name
        query = f"SELECT * FROM {NOTEBOOK_TABLE} WHERE name = $name"
        bindings = {"name": name}
        result = await db.query(query, bindings)
        
        if not result or len(result) == 0:
            raise HTTPException(status_code=404, detail=f"Notebook with name '{name}' not found")
            
        notebook = dict(result[0])
        notebook_id = notebook['id']
        if hasattr(notebook_id, 'table_name') and hasattr(notebook_id, 'record_id'):
            notebook_id = f"{notebook_id.table_name}:{notebook_id.record_id}"
        else:
            notebook_id = str(notebook_id)

        # Query sources for this notebook with all necessary fields
        query = f"""
            SELECT id, title, type, status, created, updated, metadata, 
                   full_text, notebook_id, insights, embedded_chunks
            FROM {SOURCE_TABLE} 
            WHERE notebook_id = $nb_id 
            ORDER BY created DESC
        """
        bindings = {"nb_id": notebook_id}
        result = await db.query(query, bindings)
        
        print(f"Raw sources query result for notebook {notebook_id}:")
        if result:
            for source in result:
                print(f"Raw source data: {dict(source)}")
        else:
            print("No sources found in query result")

        if not result or len(result) == 0:
            return []

        sources = []
        for source in result:
            source_dict = dict(source)
            # Convert RecordID to string
            source_dict = convert_record_id_to_string(source_dict)
            
            # Handle title from metadata if not directly present
            if not source_dict.get('title'):
                source_dict['title'] = source_dict.get('metadata', {}).get('title', 'Untitled Source')

            # Create SourceSummary with all available fields
            sources.append(SourceSummary(
                id=str(source_dict.get('id', '')),
                title=str(source_dict.get('title', '')),
                type=str(source_dict.get('type', '')),
                status=str(source_dict.get('status', '')),
                created=source_dict.get('created'),
                updated=source_dict.get('updated'),
                metadata=source_dict.get('metadata', {}),
                notebook_id=str(source_dict.get('notebook_id', '')),
                insights=source_dict.get('insights', []),
                embedded_chunks=source_dict.get('embedded_chunks', 0)
            ))

        return sources

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error listing sources for notebook {name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error listing sources: {e}")

@router.get("/api/v1/notebooks/{notebook_id}/sources", response_model=List[SourceSummary])
async def list_sources_for_notebook(
    notebook_id: str,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Lists all sources associated with a specific notebook."""
    if ":" not in notebook_id:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid notebook ID format.")
    try:
        notebook = await db.select(notebook_id)
        if not notebook:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Notebook 	{notebook_id}	 not found")

        query = f"SELECT id, title, type, status, created FROM {SOURCE_TABLE} WHERE notebook_id = $nb_id ORDER BY created DESC"
        bindings = {"nb_id": notebook_id}
        result = await db.query(query, bindings)

        sources = result[0]["result"] if result and result[0]["result"] is not None else []
        # Need to handle 'title' potentially being in metadata for non-text sources
        for source in sources:
            if not source.get('title'):
                 source['title'] = source.get('metadata', {}).get('title', 'Untitled Source')

        return sources

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error listing sources for notebook {notebook_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error listing sources")

@router.get("/api/v1/sources/{source_id}", response_model=SourceResponse)
async def get_source(
    source_id: str,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Gets full details of a specific source by its ID."""
    if ":" not in source_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid source ID format.")
    try:
        # Query source with all fields including insights and embeddings
        query = f"""
            SELECT id, title, type, status, created, updated, metadata,
                   full_text, notebook_id, insights, embedded_chunks
            FROM {SOURCE_TABLE}
            WHERE id = $source_id
        """
        bindings = {"source_id": source_id}
        result = await db.query(query, bindings)
        
        if not result or len(result) == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Source {source_id} not found")
            
        source_dict = dict(result[0])
        source_dict = convert_record_id_to_string(source_dict)
        
        # Handle title from metadata if not directly present
        if not source_dict.get('title'):
            source_dict['title'] = source_dict.get('metadata', {}).get('title', 'Untitled Source')
            
        # Create full source response
        return SourceResponse(
            id=str(source_dict.get('id', '')),
            title=str(source_dict.get('title', '')),
            type=str(source_dict.get('type', '')),
            status=str(source_dict.get('status', '')),
            created=source_dict.get('created'),
            updated=source_dict.get('updated'),
            metadata=source_dict.get('metadata', {}),
            full_text=source_dict.get('full_text', ''),
            notebook_id=str(source_dict.get('notebook_id', '')),
            insights=source_dict.get('insights', []),
            embedded_chunks=source_dict.get('embedded_chunks', 0)
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error getting source {source_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error getting source: {e}")

@router.delete("/api/v1/sources/{source_id}", response_model=StatusResponse)
async def delete_source(
    source_id: str,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Deletes a source. Note: Does not currently delete associated insights/embeddings."""
    if ":" not in source_id:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid source ID format.")
    try:
        existing = await db.select(source_id)
        if not existing:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Source 	{source_id}	 not found for deletion")

        await db.delete(source_id)
        # TODO: Add logic to delete associated insights, embeddings, potentially files
        return StatusResponse(status="success", message=f"Source {source_id} deleted successfully.")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error deleting source {source_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error deleting source")

# Placeholder for task status endpoint
# GET /api/v1/tasks/{task_id}

def convert_record_id_to_string(data: Any) -> Any:
    """Convert SurrealDB RecordID objects to strings in the response data."""
    if isinstance(data, dict):
        return {k: convert_record_id_to_string(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_record_id_to_string(item) for item in data]
    elif hasattr(data, 'table_name') and hasattr(data, 'record_id'):
        return f"{data.table_name}:{data.record_id}"
    return data

