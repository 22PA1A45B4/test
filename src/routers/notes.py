from fastapi import APIRouter, Depends, HTTPException, Query, status, Body
from typing import List, Optional, Dict
from datetime import datetime

from surrealdb import AsyncSurreal
from ..database import get_db_connection

from open_notebook.domain.models import model_manager
from open_notebook.domain.notebook import Note, Notebook
from ..models import (
    NoteCreate, NoteUpdate, NoteSummary, StatusResponse, NoteResponse
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
            "note_type": "human"
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

@router.get("/notebooks/by-name/{notebook_name}", response_model=List[NoteSummary])
async def list_notes_by_notebook_name(
    notebook_name: str,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Lists all notes in a notebook specified by name."""
    try:
        print(f"Listing notes for notebook: {notebook_name}")
        
        # First get the notebook by name
        query = f"SELECT * FROM notebook WHERE name = $name"
        bindings = {"name": notebook_name}
        result = await db.query(query, bindings)
        print(f"Notebook query result: {result}")
        
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
        print(f"Found notebook with ID: {notebook_id}")

        # Try a simpler query first
        try:
            # First try to get all notes
            all_notes_query = "SELECT * FROM note"
            all_notes = await db.query(all_notes_query)
            print(f"All notes in database: {all_notes}")
            
            # Then filter for our notebook
            notes_query = f"SELECT * FROM note WHERE notebook_id = $nb_id"
            notes_result = await db.query(notes_query, {"nb_id": notebook_id})
            print(f"Raw notes query result for notebook {notebook_id}:")
            for note in notes_result:
                print(f"Raw note data: {dict(note)}")
            
        except Exception as query_error:
            print(f"Error executing notes query: {query_error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database query error: {str(query_error)}"
            )
        
        if not notes_result:
            print("No notes found in query result")
            return []  # Return empty list instead of result
            
        notes = []
        # Handle both single result and list of results
        notes_data = notes_result[0] if isinstance(notes_result, list) and len(notes_result) > 0 else notes_result
        print(f"Initial notes_data: {notes_data}")
        
        # If notes_data is a dict with a 'result' key (SurrealDB format)
        if isinstance(notes_data, dict) and 'result' in notes_data:
            notes_data = notes_data['result']
            print(f"Extracted notes from result key: {notes_data}")
            
        # Ensure we have a list to iterate over
        if not isinstance(notes_data, list):
            notes_data = [notes_data] if notes_data else []
            print(f"Converted to list: {notes_data}")
            
        for note in notes_data:
            note_dict = dict(note)
            print(f"Processing note: {note_dict}")
            # Convert note id to string if it's a RecordID
            if hasattr(note_dict.get('id', None), 'table_name') and hasattr(note_dict.get('id', None), 'record_id'):
                note_dict['id'] = f"{note_dict['id'].table_name}:{note_dict['id'].record_id}"
            elif note_dict.get('id', None) is not None:
                note_dict['id'] = str(note_dict['id'])
                
            # Ensure all required fields are present with defaults
            note_summary = NoteSummary(
                id=str(note_dict.get("id", "")),
                title=str(note_dict.get("title", "")),
                created=note_dict.get("created", datetime.utcnow()),
                updated=note_dict.get("updated", datetime.utcnow()),
                note_type=str(note_dict.get("note_type", "human")),
            )
            notes.append(note_summary)
            
        print(f"Final notes list: {notes}")
        return notes

    except HTTPException as http_exc:
        print(f"HTTP Exception: {http_exc}")
        raise http_exc
    except Exception as e:
        print(f"Error listing notes for notebook {notebook_name}: {e}")
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

@router.delete("/{note_id}", 
    response_model=StatusResponse,
    summary="Delete a note by ID",
    description="Deletes a specific note using its ID. The ID must be in the format 'note:id'.",
    responses={
        400: {"description": "Invalid note ID format"},
        404: {"description": "Note not found"},
        500: {"description": "Internal server error"}
    }
)
async def delete_note(
    note_id: str,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Deletes a note by its ID."""
    if ":" not in note_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid note ID format. Expected table:id, got {note_id}"
        )
    try:
        # Verify note exists before deletion
        existing = await db.select(note_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Note with id {note_id} not found"
            )

        # Delete the note
        result = await db.delete(note_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete note from database"
            )

        return StatusResponse(
            status="success",
            message=f"Note {note_id} deleted successfully"
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error deleting note {note_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error deleting note: {e}"
        )

@router.delete("/by-title/{title}", 
    response_model=StatusResponse,
    summary="Delete notes by title",
    description="Deletes one or more notes by their title. Optionally scoped to a specific notebook.",
    responses={
        400: {"description": "Invalid notebook ID format"},
        404: {"description": "No notes found with the given title"},
        500: {"description": "Internal server error"}
    }
)
async def delete_note_by_title(
    title: str,
    notebook_id: Optional[str] = Query(None, description="Optional notebook ID to scope the deletion"),
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Deletes a note by its title, optionally scoped to a specific notebook."""
    try:
        # Build the query based on whether notebook_id is provided
        if notebook_id:
            if ":" not in notebook_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid notebook ID format. Expected table:id, got {notebook_id}"
                )
            # Verify notebook exists
            notebook = await db.select(notebook_id)
            if not notebook:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Notebook with id {notebook_id} not found"
                )
            query = f"DELETE {NOTE_TABLE} WHERE title = $title AND notebook_id = $notebook_id RETURN id"
            bindings = {"title": title, "notebook_id": notebook_id}
        else:
            query = f"DELETE {NOTE_TABLE} WHERE title = $title RETURN id"
            bindings = {"title": title}

        # Execute the deletion
        deleted_notes = await db.query(query, bindings)
        
        if not deleted_notes or len(deleted_notes) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No note with title '{title}' found" + 
                      (f" in notebook {notebook_id}" if notebook_id else "")
            )

        # Convert deleted notes to string IDs for the response
        deleted_ids = []
        for note in deleted_notes:
            if isinstance(note, dict) and 'id' in note:
                note_id = note['id']
                if hasattr(note_id, 'table_name') and hasattr(note_id, 'record_id'):
                    deleted_ids.append(f"{note_id.table_name}:{note_id.record_id}")
                else:
                    deleted_ids.append(str(note_id))

        return StatusResponse(
            status="success",
            message=f"Successfully deleted {len(deleted_ids)} note(s) with title '{title}'" +
                   (f" from notebook {notebook_id}" if notebook_id else "") +
                   f". Deleted IDs: {', '.join(deleted_ids)}"
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error deleting note(s) with title '{title}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error deleting note(s): {e}"
        )

@router.delete("/bulk", 
    response_model=StatusResponse,
    summary="Bulk delete notes by IDs",
    description="Deletes multiple notes by their IDs. IDs must be in the format 'note:id'.",
    responses={
        400: {"description": "No note IDs provided or invalid ID format"},
        404: {"description": "No notes found with the provided IDs"},
        500: {"description": "Internal server error"}
    }
)
async def bulk_delete_notes(
    note_ids: List[str] = Body(..., description="List of note IDs to delete"),
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Deletes multiple notes by their IDs."""
    try:
        if not note_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No note IDs provided"
            )

        # Validate all IDs have correct format
        invalid_ids = [id for id in note_ids if ":" not in id]
        if invalid_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid note ID format(s). Expected table:id, got: {', '.join(invalid_ids)}"
            )

        # Track results
        deleted_count = 0
        not_found_ids = []
        failed_ids = []

        # Process each note deletion
        for note_id in note_ids:
            try:
                # Check if note exists
                existing = await db.select(note_id)
                if not existing:
                    not_found_ids.append(note_id)
                    continue

                # Delete the note
                result = await db.delete(note_id)
                if result:
                    deleted_count += 1
                else:
                    failed_ids.append(note_id)
            except Exception as e:
                print(f"Error deleting note {note_id}: {e}")
                failed_ids.append(note_id)

        if not deleted_count:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No notes found with the provided IDs"
            )

        # Prepare response message
        message = f"Successfully deleted {deleted_count} note(s)"
        if not_found_ids:
            message += f". Notes not found: {', '.join(not_found_ids)}"
        if failed_ids:
            message += f". Failed to delete: {', '.join(failed_ids)}"

        return StatusResponse(
            status="partial_success" if (not_found_ids or failed_ids) else "success",
            message=message
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in bulk note deletion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during bulk note deletion: {e}"
        )

@router.delete("/bulk-by-title", 
    response_model=StatusResponse,
    summary="Bulk delete notes by titles",
    description="Deletes multiple notes by their titles. Optionally scoped to a specific notebook.",
    responses={
        400: {"description": "No titles provided or invalid notebook ID format"},
        404: {"description": "No notes found with the provided titles"},
        500: {"description": "Internal server error"}
    }
)
async def bulk_delete_notes_by_title(
    titles: List[str] = Body(..., description="List of note titles to delete"),
    notebook_id: Optional[str] = Query(None, description="Optional notebook ID to scope the deletion"),
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Deletes multiple notes by their titles, optionally scoped to a specific notebook."""
    try:
        if not titles:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No titles provided"
            )

        # Validate notebook_id if provided
        if notebook_id:
            if ":" not in notebook_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid notebook ID format. Expected table:id, got {notebook_id}"
                )
            # Verify notebook exists
            notebook = await db.select(notebook_id)
            if not notebook:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Notebook with id {notebook_id} not found"
                )

        # Track results
        deleted_count = 0
        not_found_titles = []
        deleted_notes = {}  # Map of title to list of deleted note IDs

        # Process each title
        for title in titles:
            try:
                # Build query based on whether notebook_id is provided
                if notebook_id:
                    query = f"DELETE {NOTE_TABLE} WHERE title = $title AND notebook_id = $notebook_id RETURN id"
                    bindings = {"title": title, "notebook_id": notebook_id}
                else:
                    query = f"DELETE {NOTE_TABLE} WHERE title = $title RETURN id"
                    bindings = {"title": title}

                # Execute deletion
                result = await db.query(query, bindings)
                
                if result and len(result) > 0:
                    # Convert note IDs to strings
                    note_ids = []
                    for note in result:
                        if isinstance(note, dict) and 'id' in note:
                            note_id = note['id']
                            if hasattr(note_id, 'table_name') and hasattr(note_id, 'record_id'):
                                note_ids.append(f"{note_id.table_name}:{note_id.record_id}")
                            else:
                                note_ids.append(str(note_id))
                    
                    if note_ids:
                        deleted_count += len(note_ids)
                        deleted_notes[title] = note_ids
                else:
                    not_found_titles.append(title)

            except Exception as e:
                print(f"Error deleting notes with title '{title}': {e}")
                not_found_titles.append(title)

        if not deleted_count:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No notes found with the provided titles" +
                      (f" in notebook {notebook_id}" if notebook_id else "")
            )

        # Prepare response message
        message = f"Successfully deleted {deleted_count} note(s)"
        if notebook_id:
            message += f" from notebook {notebook_id}"
        
        # Add details about which titles were found/not found
        if deleted_notes:
            message += ". Deleted notes:"
            for title, ids in deleted_notes.items():
                message += f"\n- '{title}': {', '.join(ids)}"
        
        if not_found_titles:
            message += f"\nNotes not found for titles: {', '.join(not_found_titles)}"

        return StatusResponse(
            status="partial_success" if not_found_titles else "success",
            message=message
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in bulk note deletion by title: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during bulk note deletion: {e}"
        )

