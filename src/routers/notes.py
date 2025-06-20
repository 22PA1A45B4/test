from fastapi import APIRouter, Depends, HTTPException, Query, status, Body
from typing import List, Optional, Dict
from datetime import datetime
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from surrealdb import AsyncSurreal
from ..database import get_db_connection

from open_notebook.domain.models import model_manager
from open_notebook.domain.notebook import Note, Notebook
from ..models import (
    NoteCreate, NoteUpdate, NoteSummary, StatusResponse, NoteResponse, NotesWithLogsResponse
)

# Create a router for note-related endpoints with proper prefix and tags
router = APIRouter(
    prefix="/api/v1/notes",
    tags=["Notes"],
    responses={
        404: {"description": "Note or Notebook not found"},
        500: {"description": "Internal server error"}
    }
)

# Define the table names for SurrealDB
NOTE_TABLE = "note"
NOTEBOOK_TABLE = "notebook"

def convert_record_id_to_string(data):
    """Convert SurrealDB RecordID objects to strings in the response data."""
    if isinstance(data, dict):
        return {k: convert_record_id_to_string(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_record_id_to_string(item) for item in data]
    elif hasattr(data, 'table_name') and hasattr(data, 'record_id'):
        return f"{data.table_name}:{data.record_id}"
    return data

@router.post("/notebooks/by-name/{notebook_name}", response_model=NoteResponse, status_code=status.HTTP_201_CREATED)
async def create_note_in_notebook_by_name(
    notebook_name: str,
    note_data: NoteCreate,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Creates a new note in a notebook specified by name. Only requires title and content."""
    try:
        # First get the notebook by name
        query = f"SELECT * FROM {NOTEBOOK_TABLE} WHERE name = $name"
        bindings = {"name": notebook_name}
        result = await db.query(query, bindings)
        
        if not result or len(result) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Notebook with name '{notebook_name}' not found"
            )
            
        notebook = dict(result[0])
        notebook_id = notebook['id']
        if hasattr(notebook_id, 'table_name') and hasattr(notebook_id, 'record_id'):
            notebook_id = f"{notebook_id.table_name}:{notebook_id.record_id}"
        else:
            notebook_id = str(notebook_id)

        # Create the note with only essential fields
        data_to_create = {
            "title": note_data.title,
            "content": note_data.content,
            "created": datetime.utcnow(),
            "updated": datetime.utcnow(),
            "notebook_id": notebook_id,
            "note_type": "human",
            "embedding": []  # Always set embedding to empty list
        }

        created_notes = await db.create(NOTE_TABLE, data_to_create)
        
        if not created_notes:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create note in database"
            )

        # Convert the created note to response format
        note_dict = dict(created_notes[0] if isinstance(created_notes, list) else created_notes)
        if hasattr(note_dict.get('id', None), 'table_name') and hasattr(note_dict.get('id', None), 'record_id'):
            note_id = f"{note_dict['id'].table_name}:{note_dict['id'].record_id}"
            note_dict['id'] = note_id
        elif note_dict.get('id', None) is not None:
            note_id = str(note_dict['id'])
            note_dict['id'] = note_id
        else:
            note_id = None

        # Create the artifact relationship so Streamlit UI can see the note
        if note_id and notebook_id:
            await db.query(f"RELATE {note_id}->artifact->{notebook_id}")

        return NoteResponse(
            id=str(note_dict.get("id", "")),
            title=str(note_dict.get("title", "")),
            content=str(note_dict.get("content", "")),
            created=note_dict.get("created"),
            updated=note_dict.get("updated"),
            note_type="human",
            notebook_id=notebook_id
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error creating note in notebook {notebook_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during note creation: {e}"
        )

@router.get("/notebooks/by-name/{notebook_name}", response_model=NotesWithLogsResponse)
async def list_notes_by_notebook_name(
    notebook_name: str,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Lists all notes in a notebook specified by name."""
    logs = []
    try:
        logs.append(f"Listing notes for notebook: {notebook_name}")
        # First get the notebook by name
        query = f"SELECT * FROM notebook WHERE name = $name"
        bindings = {"name": notebook_name}
        result = await db.query(query, bindings)
        logs.append(f"Notebook query result: {result}")
        if not result or len(result) == 0:
            logs.append(f"Notebook with name '{notebook_name}' not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Notebook with name '{notebook_name}' not found"
            )
        notebook = dict(result[0])
        notebook_id = notebook['id']
        if hasattr(notebook_id, 'table_name') and hasattr(notebook_id, 'record_id'):
            notebook_id = f"{notebook_id.table_name}:{notebook_id.record_id}"
        else:
            notebook_id = str(notebook_id)
        logs.append(f"Found notebook with ID: {notebook_id}")
        # Get all notes in the database
        all_notes_query = "SELECT * FROM note"
        all_notes = await db.query(all_notes_query)
        # Remove 'embedding' from notes before logging
        all_notes_no_embedding = []
        for note in all_notes:
            note_dict = dict(note)
            if 'embedding' in note_dict:
                note_dict = {k: v for k, v in note_dict.items() if k != 'embedding'}
            all_notes_no_embedding.append(note_dict)
        logs.append(f"All notes in database: {all_notes_no_embedding}")
        # Check for orphaned notes (notes without notebook_id)
        orphaned_notes = []
        for note in all_notes:
            note_dict = dict(note)
            if 'notebook_id' not in note_dict or not note_dict['notebook_id']:
                orphaned_notes.append(note_dict)
        # Link orphaned notes to this notebook
        if orphaned_notes:
            logs.append(f"Found {len(orphaned_notes)} orphaned notes, linking to notebook {notebook_id}")
            for note in orphaned_notes:
                note_id = note['id']
                if hasattr(note_id, 'table_name') and hasattr(note_id, 'record_id'):
                    note_id = f"{note_id.table_name}:{note_id.record_id}"
                else:
                    note_id = str(note_id)
                # Ensure embedding is set to [] if missing
                update_data = {"notebook_id": notebook_id}
                if 'embedding' not in note or note['embedding'] is None:
                    update_data["embedding"] = []
                await db.merge(note_id, update_data)
                logs.append(f"Linked note {note_id} to notebook {notebook_id}")
        # Now get notes for this notebook
        notes_query = f"SELECT * FROM note WHERE notebook_id = $nb_id"
        notes_result = await db.query(notes_query, {"nb_id": notebook_id})
        # Remove 'embedding' from notes before logging
        notes_result_no_embedding = []
        for note in notes_result:
            note_dict = dict(note)
            if 'embedding' in note_dict:
                note_dict = {k: v for k, v in note_dict.items() if k != 'embedding'}
            notes_result_no_embedding.append(note_dict)
        if notes_result:
            logs.append(f"Notes for notebook {notebook_id}: {notes_result_no_embedding}")
        else:
            return NotesWithLogsResponse(notes=[], logs=logs)
        notes = []
        for note in notes_result:
            note_dict = dict(note)
            if hasattr(note_dict.get('id', None), 'table_name') and hasattr(note_dict.get('id', None), 'record_id'):
                note_dict['id'] = f"{note_dict['id'].table_name}:{note_dict['id'].record_id}"
            elif note_dict.get('id', None) is not None:
                note_dict['id'] = str(note_dict['id'])
            note_summary = NoteSummary(
                id=str(note_dict.get("id", "")),
                title=str(note_dict.get("title", "")),
                created=note_dict.get("created", datetime.utcnow()),
                updated=note_dict.get("updated", datetime.utcnow()),
                note_type=str(note_dict.get("note_type", "human")),
            )
            notes.append(note_summary)
        logs.append(f"Returning {len(notes)} notes")
        return NotesWithLogsResponse(notes=notes, logs=logs)
    except HTTPException as http_exc:
        logs.append(f"HTTP Exception: {http_exc}")
        raise http_exc
    except Exception as e:
        logs.append(f"Error listing notes for notebook {notebook_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error listing notes: {e}"
        )

@router.post("/{note_id}/move/notebooks/by-name/{notebook_name}", response_model=NoteResponse)
async def move_note_to_notebook_by_name(
    note_id: str,
    notebook_name: str,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Moves a note to a different notebook specified by name."""
    try:
        # First get the note
        if ":" not in note_id:
            note_id = f"note:{note_id}"
        note_result = await db.select(note_id)
        if not note_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Note with id {note_id} not found"
            )

        # Get the target notebook
        query = f"SELECT * FROM notebook WHERE name = $name"
        bindings = {"name": notebook_name}
        notebook_result = await db.query(query, bindings)
        
        if not notebook_result or len(notebook_result) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Notebook with name '{notebook_name}' not found"
            )
            
        notebook = dict(notebook_result[0])
        notebook_id = notebook['id']
        if hasattr(notebook_id, 'table_name') and hasattr(notebook_id, 'record_id'):
            notebook_id = f"{notebook_id.table_name}:{notebook_id.record_id}"
        else:
            notebook_id = str(notebook_id)

        # Update the note's notebook_id
        updated_note = await db.merge(note_id, {
            "notebook_id": notebook_id,
            "updated": datetime.utcnow()
        })

        if not updated_note:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update note in database"
            )

        # Convert the updated note to response format
        note_dict = dict(updated_note)
        if hasattr(note_dict.get('id', None), 'table_name') and hasattr(note_dict.get('id', None), 'record_id'):
            note_dict['id'] = f"{note_dict['id'].table_name}:{note_dict['id'].record_id}"
        elif note_dict.get('id', None) is not None:
            note_dict['id'] = str(note_dict['id'])

        return NoteResponse(
            id=str(note_dict.get("id", "")),
            title=str(note_dict.get("title", "")),
            content=str(note_dict.get("content", "")),
            created=note_dict.get("created"),
            updated=note_dict.get("updated"),
            note_type=str(note_dict.get("note_type", "human")),
            metadata=note_dict.get("metadata", {}),
            notebook_id=notebook_id
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error moving note {note_id} to notebook {notebook_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error moving note: {e}"
        )

# Keep existing endpoints for backward compatibility and direct ID operations
@router.post("", response_model=NoteResponse, status_code=status.HTTP_201_CREATED)
async def create_note(
    note_data: NoteCreate,
    notebook_id: Optional[str] = None,
):
    """Creates a new note, optionally associated with a notebook by ID."""
    try:
        note = Note(
            title=note_data.title,
            content=note_data.content,
            note_type=note_data.note_type
        )
        note.save()
        if notebook_id:
            notebook = Notebook.get(notebook_id)
            if not notebook:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Notebook with id {notebook_id} not found"
                )
            note.add_to_notebook(notebook.id)
        return NoteResponse(**note.model_dump())

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error creating note: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during note creation: {e}"
        )

@router.get("", response_model=List[NoteSummary])
async def list_notes(
    notebook_id: Optional[str] = None,
    sort_by: Optional[str] = Query("updated", description="Field to sort by"),
    order: Optional[str] = Query("desc", description="Sort order: asc or desc"),
):
    """Lists all notes, optionally filtered by notebook_id."""
    try:
        if notebook_id:
            notebook = Notebook.get(notebook_id)
            if not notebook:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Notebook with id {notebook_id} not found"
                )
            notes = notebook.notes
        else:
            notes = Note.get_all()

        # Sorting logic (needs to be applied after fetching all notes if not handled by DB query)
        # For now, assuming Note.get_all() or notebook.notes returns a sortable list
        # and we'll handle basic sorting here if needed.
        # The domain model doesn't expose direct sorting parameters for get_all or notes property.
        # If advanced sorting is needed, it should be implemented in the domain model or repository.
        
        # Simple in-memory sort for demonstration, assuming 'updated' and 'title' are attributes
        if sort_by and hasattr(notes[0], sort_by) if notes else False:
            reverse_sort = True if order.lower() == "desc" else False
            notes.sort(key=lambda x: getattr(x, sort_by), reverse=reverse_sort)

        return [NoteSummary(**note.model_dump()) for note in notes]

    except Exception as e:
        print(f"Error listing notes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error listing notes: {e}"
        )

@router.get("/{note_id}", response_model=NoteResponse)
async def get_note(
    note_id: str,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Gets details of a specific note by its ID."""
    if ":" not in note_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid note ID format. Expected table:id, got {note_id}"
        )
    try:
        result = await db.select(note_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Note with id {note_id} not found"
            )
        
        # Convert the note to response format
        note_dict = dict(result)
        if hasattr(note_dict.get('id', None), 'table_name') and hasattr(note_dict.get('id', None), 'record_id'):
            note_dict['id'] = f"{note_dict['id'].table_name}:{note_dict['id'].record_id}"
        elif note_dict.get('id', None) is not None:
            note_dict['id'] = str(note_dict['id'])

        return NoteResponse(
            id=str(note_dict.get("id", "")),
            title=str(note_dict.get("title", "")),
            content=str(note_dict.get("content", "")),
            created=note_dict.get("created"),
            updated=note_dict.get("updated"),
            note_type="human",
            notebook_id=note_dict.get("notebook_id")
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error getting note {note_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error getting note: {e}"
        )

@router.patch("/{note_id}", response_model=NoteResponse)
async def update_note(
    note_id: str,
    note_update: NoteUpdate,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Updates a note's title, content, or other fields."""
    if ":" not in note_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid note ID format. Expected table:id, got {note_id}"
        )

    update_data = note_update.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields provided for update"
        )

    # Remove any embedding-related fields from update data
    update_data.pop('embedding', None)
    update_data.pop('needs_embedding', None)
    
    update_data["updated"] = datetime.utcnow()

    try:
        updated_note = await db.merge(note_id, update_data)
        if not updated_note:
            existing = await db.select(note_id)
            if not existing:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Note with id {note_id} not found"
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update note"
            )

        # Convert the updated note to response format, excluding embedding
        note_dict = dict(updated_note)
        if hasattr(note_dict.get('id', None), 'table_name') and hasattr(note_dict.get('id', None), 'record_id'):
            note_dict['id'] = f"{note_dict['id'].table_name}:{note_dict['id'].record_id}"
        elif note_dict.get('id', None) is not None:
            note_dict['id'] = str(note_dict['id'])

        # Remove embedding field if it exists
        note_dict.pop('embedding', None)
        note_dict.pop('needs_embedding', None)

        return NoteResponse(
            id=str(note_dict.get("id", "")),
            title=str(note_dict.get("title", "")),
            content=str(note_dict.get("content", "")),
            created=note_dict.get("created"),
            updated=note_dict.get("updated"),
            note_type=str(note_dict.get("note_type", "human")),
            metadata=note_dict.get("metadata", {}),
            notebook_id=note_dict.get("notebook_id")
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error updating note {note_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error updating note: {e}"
        )

