from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, List, Literal, Optional
from datetime import datetime
import os

from surrealdb import AsyncSurreal
from pydantic import BaseModel, ConfigDict

from ..database import get_db_connection
from ..models import Model, ModelCreate, ModelUpdate, DefaultModels, StatusResponse

router = APIRouter(
    prefix="/api/v1",
    tags=["Models"],
)

MODEL_TABLE = "model"
DEFAULT_MODELS_RECORD = "open_notebook:default_models"

# --- Models for API ---
class ProviderStatus(BaseModel):
    available: List[str]
    unavailable: List[str]
    model_config = ConfigDict(from_attributes=True)

class ModelType(BaseModel):
    type: str
    available: bool
    model_config = ConfigDict(from_attributes=True)

class ModelWithProvider(Model):
    provider_status: bool
    model_config = ConfigDict(from_attributes=True)

# --- Helper Functions ---
def get_provider_status() -> Dict[str, bool]:
    """Get the status of all providers based on environment variables"""
    status = {}
    status["ollama"] = os.environ.get("OLLAMA_API_BASE") is not None
    status["openai"] = os.environ.get("OPENAI_API_KEY") is not None
    status["groq"] = os.environ.get("GROQ_API_KEY") is not None
    status["xai"] = os.environ.get("XAI_API_KEY") is not None
    status["vertexai"] = (
        os.environ.get("VERTEX_PROJECT") is not None
        and os.environ.get("VERTEX_LOCATION") is not None
        and os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is not None
    )
    status["vertexai-anthropic"] = (
        os.environ.get("VERTEX_PROJECT") is not None
        and os.environ.get("VERTEX_LOCATION") is not None
        and os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is not None
    )
    status["gemini"] = os.environ.get("GEMINI_API_KEY") is not None
    status["openrouter"] = (
        os.environ.get("OPENROUTER_API_KEY") is not None
        and os.environ.get("OPENAI_API_KEY") is not None
        and os.environ.get("OPENROUTER_BASE_URL") is not None
    )
    status["anthropic"] = os.environ.get("ANTHROPIC_API_KEY") is not None
    status["elevenlabs"] = os.environ.get("ELEVENLABS_API_KEY") is not None
    status["litellm"] = (
        status["ollama"]
        or status["vertexai"]
        or status["vertexai-anthropic"]
        or status["anthropic"]
        or status["openai"]
        or status["gemini"]
    )
    return status

# --- Provider Status Endpoint ---
@router.get("/models/providers", response_model=ProviderStatus)
async def get_providers():
    """Get the status of all model providers"""
    status = get_provider_status()
    return ProviderStatus(
        available=[k for k, v in status.items() if v],
        unavailable=[k for k, v in status.items() if not v]
    )

# --- Model Type Endpoints ---
@router.get("/models/types", response_model=List[ModelType])
async def get_model_types(db: AsyncSurreal = Depends(get_db_connection)):
    """Get all model types and their availability"""
    model_types = [
        "language",
        "embedding",
        "text_to_speech",
        "speech_to_text"
    ]
    
    # Check which types have models configured
    types_with_models = set()
    query = f"SELECT type FROM {MODEL_TABLE}"
    result = await db.query(query)
    if result:
        types_with_models = {model["type"] for model in result}
    
    return [
        ModelType(type=type_, available=type_ in types_with_models)
        for type_ in model_types
    ]

# --- Model CRUD Endpoints ---
@router.post("/models", response_model=Model, status_code=status.HTTP_201_CREATED)
async def create_model(
    model: ModelCreate,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Create a new model configuration"""
    # Verify provider is available
    provider_status = get_provider_status()
    if not provider_status.get(model.provider, False):
        raise HTTPException(
            status_code=400,
            detail=f"Provider {model.provider} is not available. Please check your environment variables."
        )
    
    # Create the model
    data = model.model_dump()
    data["created"] = datetime.utcnow()
    data["updated"] = datetime.utcnow()
    
    created = await db.create(MODEL_TABLE, data)
    if not created:
        raise HTTPException(status_code=500, detail="Failed to create model")
    return Model(**created[0])

@router.get("/models", response_model=List[ModelWithProvider])
async def list_models(
    type: Optional[str] = None,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """List all models, optionally filtered by type"""
    provider_status = get_provider_status()
    query = f"SELECT * FROM {MODEL_TABLE}"
    if type:
        query += " WHERE type = $type"
        result = await db.query(query, {"type": type})
    else:
        result = await db.query(query)
    
    if not result:
        return []
    
    # Add provider status to each model
    models = []
    for model in result:
        model_data = dict(model)
        model_data["provider_status"] = provider_status.get(model["provider"], False)
        models.append(ModelWithProvider(**model_data))
    
    return models

@router.get("/models/{model_id}", response_model=ModelWithProvider)
async def get_model(
    model_id: str,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Get a specific model by ID"""
    if ":" not in model_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid model ID format. Expected table:id"
        )
    
    model = await db.select(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    provider_status = get_provider_status()
    model_data = dict(model)
    model_data["provider_status"] = provider_status.get(model["provider"], False)
    return ModelWithProvider(**model_data)

@router.patch("/models/{model_id}", response_model=Model)
async def update_model(
    model_id: str,
    model_update: ModelUpdate,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Update a model configuration"""
    if ":" not in model_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid model ID format. Expected table:id"
        )
    
    # Check if model exists
    existing = await db.select(model_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Update the model
    update_data = model_update.model_dump(exclude_unset=True)
    update_data["updated"] = datetime.utcnow()
    
    updated = await db.merge(model_id, update_data)
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update model")
    return Model(**updated)

@router.delete("/models/{model_id}", response_model=StatusResponse)
async def delete_model(
    model_id: str,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Delete a model configuration"""
    if ":" not in model_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid model ID format. Expected table:id"
        )
    
    # Check if model exists
    existing = await db.select(model_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Delete the model
    await db.delete(model_id)
    return StatusResponse(
        status="success",
        message=f"Model {model_id} deleted successfully"
    )

# --- Default Models Endpoints ---
@router.get("/models/defaults", response_model=DefaultModels)
async def get_default_models(db: AsyncSurreal = Depends(get_db_connection)):
    """Get the current default model configurations"""
    try:
        defaults = await db.select(DEFAULT_MODELS_RECORD)
        if not defaults:
            # Initialize with empty defaults if not exists
            defaults = {
                "id": DEFAULT_MODELS_RECORD,
                "default_chat_model": None,
                "default_transformation_model": None,
                "large_context_model": None,
                "default_text_to_speech_model": None,
                "default_speech_to_text_model": None,
                "default_embedding_model": None,
                "default_tools_model": None,
                "created": datetime.utcnow(),
                "updated": datetime.utcnow()
            }
            try:
                created = await db.create(DEFAULT_MODELS_RECORD, defaults)
                if not created:
                    raise HTTPException(status_code=500, detail="Failed to initialize default models")
                defaults = created[0]
            except Exception as create_error:
                print(f"Error creating default models: {create_error}")
                # If creation fails, return empty defaults
                return DefaultModels(
                    id=DEFAULT_MODELS_RECORD,
                    created=datetime.utcnow(),
                    updated=datetime.utcnow()
                )
        
        # Handle both single record and list responses from SurrealDB
        if isinstance(defaults, list):
            defaults = defaults[0]
        
        # Convert to dict if not already
        if not isinstance(defaults, dict):
            defaults = dict(defaults)
        
        # Ensure id is set
        if 'id' not in defaults:
            defaults['id'] = DEFAULT_MODELS_RECORD
        
        # Remove any extra fields that aren't in our model
        model_fields = DefaultModels.model_fields.keys()
        filtered_defaults = {k: v for k, v in defaults.items() if k in model_fields}
        
        return DefaultModels(**filtered_defaults)
    except Exception as e:
        print(f"Error retrieving default models: {e}")
        # Return empty defaults on error
        return DefaultModels(
            id=DEFAULT_MODELS_RECORD,
            created=datetime.utcnow(),
            updated=datetime.utcnow()
        )

@router.patch("/models/defaults", response_model=DefaultModels)
async def update_default_models(
    defaults: DefaultModels,
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Update the default model configurations"""
    try:
        # First check if the record exists
        existing = await db.select(DEFAULT_MODELS_RECORD)
        if not existing:
            # If it doesn't exist, create it
            data = defaults.model_dump()
            data["id"] = DEFAULT_MODELS_RECORD
            data["created"] = datetime.utcnow()
            data["updated"] = datetime.utcnow()
            created = await db.create(DEFAULT_MODELS_RECORD, data)
            if not created:
                raise HTTPException(status_code=500, detail="Failed to create default models")
            return DefaultModels(**created[0])
        
        # Update existing record
        update_data = defaults.model_dump(exclude_unset=True)
        update_data["updated"] = datetime.utcnow()
        
        updated = await db.merge(DEFAULT_MODELS_RECORD, update_data)
        if not updated:
            raise HTTPException(status_code=500, detail="Failed to update default models")
        
        # Handle both single record and list responses
        if isinstance(updated, list):
            updated = updated[0]
            
        # Filter fields to match our model
        model_fields = DefaultModels.model_fields.keys()
        filtered_updated = {k: v for k, v in updated.items() if k in model_fields}
        
        return DefaultModels(**filtered_updated)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating default models: {str(e)}"
        ) 