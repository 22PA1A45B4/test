# /home/ubuntu/open_notebook_full_backend/fastapi_backend/src/routers/podcasts.py

from fastapi import (
    APIRouter, Depends, HTTPException, status
)
from typing import List, Optional, Union, Dict, Any
from datetime import datetime

from surrealdb import AsyncSurreal

from ..database import get_db_connection
# Import necessary models - add specific request/response models as needed
from ..models import (
    StatusResponse, TaskStatus # Add PodcastEpisode, PodcastConfig etc. later
)

# Placeholder models (should be in models.py)
from pydantic import BaseModel, ConfigDict

class PodcastConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    name: str
    podcast_name: Optional[str] = None
    podcast_tagline: Optional[str] = None
    output_language: str = "English"
    person1_role: List[str] = []
    person2_role: List[str] = []
    conversation_style: List[str] = []
    engagement_technique: List[str] = []
    dialogue_structure: List[str] = []
    transcript_model: Optional[str] = None
    transcript_model_provider: Optional[str] = None
    user_instructions: Optional[str] = None
    ending_message: Optional[str] = None
    creativity: float = 0.7
    provider: str = "openai"
    voice1: str
    voice2: str
    model: str
    created: Optional[datetime] = None
    updated: Optional[datetime] = None

class PodcastConfigCreate(BaseModel):
    name: str
    podcast_name: Optional[str] = None
    podcast_tagline: Optional[str] = None
    output_language: str = "English"
    person1_role: List[str] = []
    person2_role: List[str] = []
    conversation_style: List[str] = []
    engagement_technique: List[str] = []
    dialogue_structure: List[str] = []
    transcript_model: Optional[str] = None
    transcript_model_provider: Optional[str] = None
    user_instructions: Optional[str] = None
    ending_message: Optional[str] = None
    creativity: float = 0.7
    provider: str = "openai"
    voice1: str
    voice2: str
    model: str

class PodcastConfigUpdate(BaseModel):
    name: Optional[str] = None
    podcast_name: Optional[str] = None
    podcast_tagline: Optional[str] = None
    output_language: Optional[str] = None
    person1_role: Optional[List[str]] = None
    person2_role: Optional[List[str]] = None
    conversation_style: Optional[List[str]] = None
    engagement_technique: Optional[List[str]] = None
    dialogue_structure: Optional[List[str]] = None
    transcript_model: Optional[str] = None
    transcript_model_provider: Optional[str] = None
    user_instructions: Optional[str] = None
    ending_message: Optional[str] = None
    creativity: Optional[float] = None
    provider: Optional[str] = None
    voice1: Optional[str] = None
    voice2: Optional[str] = None
    model: Optional[str] = None

class PodcastEpisodeSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    name: Optional[str] = None
    template: str
    template_name: Optional[str] = None
    notebook_name: Optional[str] = None
    created: datetime
    status: str = "pending"

class PodcastEpisode(PodcastEpisodeSummary):
    audio_url: Optional[str] = None
    source_text: Optional[str] = None
    instructions: Optional[str] = None

# Define content source references for generation request
class NotebookRef(BaseModel):
    type: str = "notebook"
    id: str

class SourceRef(BaseModel):
    type: str = "source"
    id: str

class NoteRef(BaseModel):
    type: str = "note"
    id: str

class NotebookNameRef(BaseModel):
    type: str = "notebook_name"
    name: str

class TemplateNameRef(BaseModel):
    type: str = "template_name"
    name: str

class PodcastGenerateRequest(BaseModel):
    template_name: str
    notebook_name: str
    episode_name: Optional[str] = None

class NotebookInfo(BaseModel):
    id: str
    name: str

class TemplateInfo(BaseModel):
    id: str
    name: str

# Create a router for podcast-related endpoints
router = APIRouter(
    prefix="/api/v1/podcasts",
    tags=["Podcasts"],
)

PODCAST_CONFIG_TABLE = "podcast_config"
PODCAST_EPISODE_TABLE = "podcast_episode"

def convert_record_id_to_string(data: Any) -> Any:
    """Convert SurrealDB RecordID objects to strings in the response data."""
    if isinstance(data, dict):
        return {k: convert_record_id_to_string(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_record_id_to_string(item) for item in data]
    elif hasattr(data, 'table_name') and hasattr(data, 'record_id'):
        return f"{data.table_name}:{data.record_id}"
    return data

# --- Podcast Generation ---

@router.post("/generate", response_model=TaskStatus, status_code=status.HTTP_202_ACCEPTED)
async def generate_podcast_episode(
    request_data: PodcastGenerateRequest,
    db: AsyncSurreal = Depends(get_db_connection)
) -> TaskStatus:
    """Triggers the generation of a new podcast episode based on a template and notebook name."""
    try:
        print(f"Starting podcast generation with template: {request_data.template_name} and notebook: {request_data.notebook_name}")
        
        # Find template by name
        template_query = f"SELECT * FROM {PODCAST_CONFIG_TABLE} WHERE name = $name"
        template_bindings = {"name": request_data.template_name}
        template_result = await db.query(template_query, template_bindings)
        print(f"Template query result: {template_result}")
        
        if not template_result or len(template_result) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template with name '{request_data.template_name}' not found"
            )
        
        template = template_result[0]
        template_id = template['id']
        if hasattr(template_id, 'table_name') and hasattr(template_id, 'record_id'):
            template_id = f"{template_id.table_name}:{template_id.record_id}"
        print(f"Found template ID: {template_id}")
        
        # Find notebook by name
        notebook_query = "SELECT * FROM notebook WHERE name = $name"
        notebook_bindings = {"name": request_data.notebook_name}
        notebook_result = await db.query(notebook_query, notebook_bindings)
        print(f"Notebook query result: {notebook_result}")
        
        if not notebook_result or len(notebook_result) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Notebook with name '{request_data.notebook_name}' not found"
            )
        
        notebook = notebook_result[0]
        notebook_id = notebook['id']
        if hasattr(notebook_id, 'table_name') and hasattr(notebook_id, 'record_id'):
            notebook_id = f"{notebook_id.table_name}:{notebook_id.record_id}"
        print(f"Found notebook ID: {notebook_id}")
        
        # Get notebook content from related notes
        notes_query = """
            SELECT * FROM note 
            WHERE id IN (
                SELECT in FROM artifact WHERE out = $notebook_id
            )
        """
        notes_bindings = {"notebook_id": notebook_id}
        notes_result = await db.query(notes_query, notes_bindings)
        print(f"Notes query result: {notes_result}")
        
        if not notes_result or len(notes_result) == 0:
            # Try alternative query to check if notes exist
            check_notes_query = """
                SELECT * FROM note 
                WHERE notebook_id = $notebook_id
            """
            check_result = await db.query(check_notes_query, notes_bindings)
            print(f"Check notes query result: {check_result}")
            
            if not check_result or len(check_result) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Notebook has no notes to generate podcast from. Please add some notes to your notebook first."
                )
            else:
                notes_result = check_result
        
        # Combine all note contents
        notebook_content = "\n\n".join([
            f"--- {note.get('title', 'Untitled Note')} ---\n{note.get('content', '')}"
            for note in notes_result
            if note.get('content')
        ])
        
        if not notebook_content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Notebook notes have no content to generate podcast from. Please add content to your notes."
            )
        
        print(f"Retrieved notebook content length: {len(notebook_content)}")
        
        # Create new episode record
        episode_name = request_data.episode_name or f"Episode {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
        episode_data = {
            "name": episode_name,
            "template": template_id,
            "template_name": template['name'],
            "notebook_name": notebook['name'],
            "created": datetime.utcnow().isoformat(),
            "status": "pending",
            "text": notebook_content,
            "instructions": template.get('user_instructions', ''),
            "content_source": {
                "type": "notebook",
                "id": notebook_id
            }
        }
        print(f"Creating episode with data: {episode_data}")
        
        try:
            # Create the episode record
            result = await db.create(PODCAST_EPISODE_TABLE, episode_data)
            print(f"Create episode result: {result}")
            
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create podcast episode: No result returned"
                )
            
            # Get the created episode ID
            episode_id = result[0]['id']
            if hasattr(episode_id, 'table_name') and hasattr(episode_id, 'record_id'):
                episode_id = f"{episode_id.table_name}:{episode_id.record_id}"
            print(f"Created episode ID: {episode_id}")
            
            # Create a task ID for tracking
            task_id = f"task_podcast_{episode_id}"
            
            # Start podcast generation in background
            try:
                from podcastfy.client import generate_podcast
                
                # Prepare conversation config
                conversation_config = {
                    "max_num_chunks": 8,  # Default to longer format
                    "min_chunk_size": 600,
                    "conversation_style": template.get('conversation_style', []),
                    "roles_person1": template.get('person1_role', []),
                    "roles_person2": template.get('person2_role', []),
                    "dialogue_structure": template.get('dialogue_structure', []),
                    "podcast_name": template.get('podcast_name', ''),
                    "podcast_tagline": template.get('podcast_tagline', ''),
                    "output_language": template.get('output_language', 'English'),
                    "user_instructions": template.get('user_instructions', ''),
                    "engagement_techniques": template.get('engagement_technique', []),
                    "creativity": template.get('creativity', 0.7),
                    "text_to_speech": {
                        "output_directories": {
                            "transcripts": "/data/podcasts/transcripts",
                            "audio": "/data/podcasts/audio",
                        },
                        "temp_audio_dir": "/data/podcasts/audio/tmp",
                        "ending_message": template.get('ending_message', 'Thank you for listening to this episode. Don\'t forget to subscribe to our podcast for more interesting conversations.'),
                        "default_tts_model": template.get('provider', 'openai'),
                        template.get('provider', 'openai'): {
                            "default_voices": {
                                "question": template.get('voice1', ''),
                                "answer": template.get('voice2', '')
                            },
                            "model": template.get('model', '')
                        },
                        "audio_format": "mp3",
                    },
                }
                
                # Determine API key and model settings
                api_key_label = None
                llm_model_name = None
                tts_model = None
                
                if template.get('transcript_model_provider'):
                    if template['transcript_model_provider'] == "openai":
                        api_key_label = "OPENAI_API_KEY"
                        llm_model_name = template.get('transcript_model')
                    elif template['transcript_model_provider'] == "anthropic":
                        api_key_label = "ANTHROPIC_API_KEY"
                        llm_model_name = template.get('transcript_model')
                    elif template['transcript_model_provider'] == "gemini":
                        api_key_label = "GEMINI_API_KEY"
                        llm_model_name = template.get('transcript_model')
                
                if template.get('provider') == "gemini":
                    tts_model = "geminimulti"
                elif template.get('provider') == "openai":
                    tts_model = "openai"
                elif template.get('provider') == "anthropic":
                    tts_model = "anthropic"
                elif template.get('provider') == "elevenlabs":
                    tts_model = "elevenlabs"
                
                # Generate podcast
                audio_file = generate_podcast(
                    conversation_config=conversation_config,
                    text=notebook_content,
                    tts_model=tts_model,
                    llm_model_name=llm_model_name,
                    api_key_label=api_key_label,
                    longform=True
                )
                
                # Update episode with audio file
                await db.merge(episode_id, {
                    "audio_file": audio_file,
                    "status": "completed"
                })
                
            except Exception as gen_error:
                print(f"Error generating podcast: {gen_error}")
                await db.merge(episode_id, {
                    "status": "failed",
                    "error": str(gen_error)
                })
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error generating podcast: {str(gen_error)}"
                )
            
            return TaskStatus(
                task_id=task_id,
                status="pending",
                message=f"Podcast generation started successfully for episode: {episode_name}"
            )
            
        except Exception as db_error:
            print(f"Database error creating podcast episode: {db_error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error creating podcast episode: {str(db_error)}"
            )
            
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error generating podcast episode: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during podcast generation: {str(e)}"
        )

# --- Podcast Episodes ---

@router.get("/episodes", response_model=List[PodcastEpisodeSummary])
async def list_podcast_episodes(
    db: AsyncSurreal = Depends(get_db_connection)
) -> List[PodcastEpisodeSummary]:
    """Lists all generated podcast episodes."""
    try:
        # First get all episodes
        query = f"SELECT * FROM {PODCAST_EPISODE_TABLE} ORDER BY created DESC"
        result = await db.query(query)
        print("SurrealDB /episodes raw result:", result)
        
        episodes = []
        for ep in (result if result is not None else []):
            ep_dict = ep.model_dump() if hasattr(ep, 'model_dump') else dict(ep)
            
            # Convert RecordID to string for id
            if hasattr(ep_dict.get('id', None), 'table_name') and hasattr(ep_dict.get('id', None), 'record_id'):
                ep_dict['id'] = f"{ep_dict['id'].table_name}:{ep_dict['id'].record_id}"
            elif ep_dict.get('id', None) is not None:
                ep_dict['id'] = str(ep_dict['id'])
            
            # Convert template RecordID to string
            if hasattr(ep_dict.get('template', None), 'table_name') and hasattr(ep_dict.get('template', None), 'record_id'):
                ep_dict['template'] = f"{ep_dict['template'].table_name}:{ep_dict['template'].record_id}"
            elif ep_dict.get('template', None) is not None:
                ep_dict['template'] = str(ep_dict['template'])
            
            # Get template name
            template_id = ep_dict.get('template')
            if template_id:
                template_query = f"SELECT name FROM {PODCAST_CONFIG_TABLE} WHERE id = $id"
                template_bindings = {"id": template_id}
                template_result = await db.query(template_query, template_bindings)
                if template_result and len(template_result) > 0:
                    ep_dict['template_name'] = template_result[0].get('name')
            
            # Get notebook name from content_source
            content_source = ep_dict.get('content_source', {})
            if content_source and content_source.get('type') == 'notebook':
                notebook_id = content_source.get('id')
                if notebook_id:
                    notebook_query = "SELECT name FROM notebook WHERE id = $id"
                    notebook_bindings = {"id": notebook_id}
                    notebook_result = await db.query(notebook_query, notebook_bindings)
                    if notebook_result and len(notebook_result) > 0:
                        ep_dict['notebook_name'] = notebook_result[0].get('name')
            
            episodes.append(ep_dict)
        
        print("Episodes to return:", episodes)
        return convert_record_id_to_string(episodes)
    except Exception as e:
        print(f"Error listing podcast episodes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error listing episodes: {e}"
        )

@router.get("/episodes/{episode_id}", response_model=PodcastEpisode)
async def get_podcast_episode(
    episode_id: str,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Gets details of a specific podcast episode."""
    if ":" not in episode_id:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid episode ID format.")
    try:
        result = await db.select(episode_id)
        if isinstance(result, list):
            if result:
                episode = result[0]
            else:
                raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
        elif result:
            episode = result
        else:
            raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
        # Convert id to string if needed
        if hasattr(episode.get('id', None), 'table_name') and hasattr(episode.get('id', None), 'record_id'):
            episode['id'] = f"{episode['id'].table_name}:{episode['id'].record_id}"
        elif episode.get('id', None) is not None:
            episode['id'] = str(episode['id'])
        return convert_record_id_to_string(episode)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error getting podcast episode {episode_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error getting episode: {e}")

@router.delete("/episodes/{episode_id}", response_model=StatusResponse)
async def delete_podcast_episode(
    episode_id: str,
    db: AsyncSurreal = Depends(get_db_connection)
) -> StatusResponse:
    """Deletes a generated podcast episode by ID or name."""
    try:
        # First try to find by ID
        if ":" in episode_id:
            existing = await db.select(episode_id)
            if existing:
                await db.delete(episode_id)
                return StatusResponse(status="success", message=f"Episode {episode_id} deleted successfully.")
        
        # If not found by ID or no ID format, try to find by name
        query = f"SELECT * FROM {PODCAST_EPISODE_TABLE} WHERE name = $name"
        bindings = {"name": episode_id}
        result = await db.query(query, bindings)
        
        if not result or len(result) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Episode with ID or name '{episode_id}' not found"
            )
            
        # Delete all episodes with matching name
        for episode in result:
            episode_id = episode['id']
            if hasattr(episode_id, 'table_name') and hasattr(episode_id, 'record_id'):
                episode_id = f"{episode_id.table_name}:{episode_id.record_id}"
            await db.delete(episode_id)
            
        return StatusResponse(
            status="success",
            message=f"All episodes with name '{episode_id}' deleted successfully."
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error deleting podcast episode {episode_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error deleting episode: {e}"
        )

@router.get("/episodes/by-name/{name}", response_model=PodcastEpisode)
async def get_podcast_episode_by_name(
    name: str,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Gets details of a specific podcast episode by its name."""
    try:
        query = f"SELECT * FROM {PODCAST_EPISODE_TABLE} WHERE name = $name"
        bindings = {"name": name}
        result = await db.query(query, bindings)
        if not result or len(result) == 0:
            raise HTTPException(status_code=404, detail=f"Episode with name '{name}' not found")
        episode = dict(result[0])
        # Convert id to string if needed
        if hasattr(episode.get('id', None), 'table_name') and hasattr(episode.get('id', None), 'record_id'):
            episode['id'] = f"{episode['id'].table_name}:{episode['id'].record_id}"
        elif episode.get('id', None) is not None:
            episode['id'] = str(episode['id'])
        return convert_record_id_to_string(episode)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error getting podcast episode by name {name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error getting episode: {e}")

@router.delete("/episodes/by-name/{name}", response_model=StatusResponse)
async def delete_podcast_episode_by_name(
    name: str,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Deletes podcast episodes by name."""
    try:
        # Find all episodes with the given name
        query = f"SELECT * FROM {PODCAST_EPISODE_TABLE} WHERE name = $name"
        bindings = {"name": name}
        result = await db.query(query, bindings)
        
        if not result or len(result) == 0:
            raise HTTPException(status_code=404, detail=f"No episodes found with name '{name}'")
            
        # Delete each episode
        for episode in result:
            episode_id = episode['id']
            await db.delete(episode_id)
            
        return StatusResponse(status="success", message=f"All episodes with name '{name}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting podcast episodes with name {name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error deleting episodes: {e}")

# --- Podcast Templates ---

@router.post("/templates", response_model=PodcastConfig, status_code=status.HTTP_201_CREATED)
async def create_podcast_template(
    template_data: PodcastConfigCreate,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Creates a new podcast template."""
    try:
        # Check if template with same name already exists
        query = f"SELECT * FROM {PODCAST_CONFIG_TABLE} WHERE name = $name"
        bindings = {"name": template_data.name}
        existing = await db.query(query, bindings)
        if existing and len(existing) > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Template with name '{template_data.name}' already exists"
            )

        # Add timestamps
        data_to_create = template_data.model_dump()
        data_to_create["created"] = datetime.utcnow()
        data_to_create["updated"] = datetime.utcnow()

        result = await db.create(PODCAST_CONFIG_TABLE, data_to_create)
        if not result:
            raise HTTPException(status_code=500, detail="Failed to create podcast template")
            
        return convert_record_id_to_string(result[0])
    except Exception as e:
        print(f"Error creating podcast template: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error creating template: {e}")

@router.get("/templates", response_model=List[PodcastConfig])
async def list_podcast_templates(
    db: AsyncSurreal = Depends(get_db_connection)
) -> List[PodcastConfig]:
    """Lists all saved podcast templates."""
    try:
        query = f"SELECT * FROM {PODCAST_CONFIG_TABLE} ORDER BY updated DESC"
        result = await db.query(query)
        
        if not result:
            return []
            
        templates = []
        for tmpl in result[0]:
            tmpl_dict = tmpl.model_dump() if hasattr(tmpl, 'model_dump') else dict(tmpl)
            if hasattr(tmpl_dict.get('id', None), 'table_name') and hasattr(tmpl_dict.get('id', None), 'record_id'):
                tmpl_dict['id'] = f"{tmpl_dict['id'].table_name}:{tmpl_dict['id'].record_id}"
            elif tmpl_dict.get('id', None) is not None:
                tmpl_dict['id'] = str(tmpl_dict['id'])
            templates.append(tmpl_dict)
            
        return convert_record_id_to_string(templates)
    except Exception as e:
        print(f"Error listing podcast templates: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error listing templates: {e}")

@router.get("/templates/{template_id}", response_model=PodcastConfig)
async def get_podcast_template(
    template_id: str,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Gets details of a specific podcast template."""
    if ":" not in template_id:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid template ID format.")
    
    try:
        result = await db.select(template_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
            
        template = dict(result)
        if hasattr(template.get('id', None), 'table_name') and hasattr(template.get('id', None), 'record_id'):
            template['id'] = f"{template['id'].table_name}:{template['id'].record_id}"
        elif template.get('id', None) is not None:
            template['id'] = str(template['id'])
            
        return convert_record_id_to_string(template)
    except Exception as e:
        print(f"Error getting podcast template {template_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error getting template: {e}")

@router.patch("/templates/{template_id}", response_model=PodcastConfig)
async def update_podcast_template(
    template_id: str,
    template_update: PodcastConfigUpdate,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Updates an existing podcast template."""
    if ":" not in template_id:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid template ID format.")
    
    try:
        update_data = template_update.model_dump(exclude_unset=True)
        if not update_data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update.")

        update_data["updated"] = datetime.utcnow()
        
        result = await db.merge(template_id, update_data)
        if not result:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
            
        template = dict(result)
        if hasattr(template.get('id', None), 'table_name') and hasattr(template.get('id', None), 'record_id'):
            template['id'] = f"{template['id'].table_name}:{template['id'].record_id}"
        elif template.get('id', None) is not None:
            template['id'] = str(template['id'])
            
        return convert_record_id_to_string(template)
    except Exception as e:
        print(f"Error updating podcast template {template_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error updating template: {e}")

@router.post("/templates/{template_id}/duplicate", response_model=PodcastConfig, status_code=status.HTTP_201_CREATED)
async def duplicate_podcast_template(
    template_id: str,
    db: AsyncSurreal = Depends(get_db_connection)
) -> PodcastConfig:
    """Duplicates an existing podcast template."""
    if ":" not in template_id:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid template ID format.")
    
    try:
        # Get original template
        original = await db.select(template_id)
        if not original:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
            
        # Create copy with modified name
        template_data = original.model_dump() if hasattr(original, 'model_dump') else dict(original)
        template_data['name'] = f"{template_data['name']} - Copy"
        template_data['created'] = datetime.utcnow()
        template_data['updated'] = datetime.utcnow()
        
        result = await db.create(PODCAST_CONFIG_TABLE, template_data)
        if not result:
            raise HTTPException(status_code=500, detail="Failed to duplicate template")
            
        return convert_record_id_to_string(result[0])
    except Exception as e:
        print(f"Error duplicating podcast template {template_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error duplicating template: {e}")

@router.delete("/templates/{template_id}", response_model=StatusResponse)
async def delete_podcast_template(
    template_id: str,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Deletes a podcast template."""
    if ":" not in template_id:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid template ID format.")
    
    try:
        existing = await db.select(template_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
            
        await db.delete(template_id)
        return StatusResponse(status="success", message=f"Template {template_id} deleted successfully.")
    except Exception as e:
        print(f"Error deleting podcast template {template_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error deleting template: {e}")

@router.get("/available-notebooks", response_model=List[NotebookInfo])
async def get_available_notebooks(
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Gets a list of available notebooks for podcast generation."""
    try:
        query = "SELECT * FROM notebook WHERE archived != true"
        result = await db.query(query)
        
        if not result:
            return []
            
        notebooks = []
        for notebook in result:
            notebook = dict(notebook)
            if hasattr(notebook.get('id', None), 'table_name') and hasattr(notebook.get('id', None), 'record_id'):
                notebook['id'] = f"{notebook['id'].table_name}:{notebook['id'].record_id}"
            elif notebook.get('id', None) is not None:
                notebook['id'] = str(notebook['id'])
            notebooks.append(NotebookInfo(id=notebook['id'], name=notebook['name']))
            
        return notebooks
    except Exception as e:
        print(f"Error getting available notebooks: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error getting notebooks: {e}")

@router.get("/available-templates", response_model=List[TemplateInfo])
async def get_available_templates(
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Gets a list of available podcast templates."""
    try:
        query = f"SELECT * FROM {PODCAST_CONFIG_TABLE}"
        result = await db.query(query)
        
        if not result:
            return []
            
        templates = []
        for template in result:
            template = dict(template)
            if hasattr(template.get('id', None), 'table_name') and hasattr(template.get('id', None), 'record_id'):
                template['id'] = f"{template['id'].table_name}:{template['id'].record_id}"
            elif template.get('id', None) is not None:
                template['id'] = str(template['id'])
            templates.append(TemplateInfo(id=template['id'], name=template['name']))
            
        return templates
    except Exception as e:
        print(f"Error getting available templates: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error getting templates: {e}")