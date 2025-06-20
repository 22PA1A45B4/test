# /home/ubuntu/open_notebook_full_backend/fastapi_backend/src/routers/ai_interactions.py

from fastapi import (
    APIRouter, Depends, HTTPException, status, Header
)
from typing import List, Optional, Dict, Any
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime

from surrealdb import AsyncSurreal

from ..database import get_db_connection
# Import necessary models
from ..models import (
    StatusResponse, TaskStatus, SourceInsight, ChatResponse, AskResponse, Citation,
    ChatContext, Source, Note, ApplyTransformationRequest, ChatRequest, AskRequest
)
from open_notebook.models import MODEL_CLASS_MAP, LanguageModel
from pydantic import BaseModel

# Create a router for AI interaction endpoints
router = APIRouter(
    prefix="/api/v1",
    tags=["AI Interactions"],
)

SOURCE_TABLE = "source"
NOTE_TABLE = "note"
NOTEBOOK_TABLE = "notebook"
TRANSFORMATION_TABLE = "transformation" # Assuming table name

# Default model configuration
DEFAULT_MODEL_PROVIDER = os.getenv("DEFAULT_MODEL_PROVIDER", "openai")
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "gpt-3.5-turbo")

def get_model_class(provider: str) -> type[LanguageModel]:
    """Get the appropriate model class for the given provider."""
    if provider not in MODEL_CLASS_MAP["language"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model provider: {provider}. Supported providers: {list(MODEL_CLASS_MAP['language'].keys())}"
        )
    return MODEL_CLASS_MAP["language"][provider]

def initialize_model(provider: str, model_name: str, api_key: Optional[str] = None) -> LanguageModel:
    """Initialize a language model with the given configuration."""
    model_class = get_model_class(provider)
    
    # Set API key in environment if provided
    if api_key:
        if provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
        elif provider == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = api_key
        elif provider == "groq":
            os.environ["GROQ_API_KEY"] = api_key
        # Add other providers as needed
    
    # Initialize model with appropriate parameters
    model_params = {
        "model_name": model_name,
        "temperature": 0.7,
        "max_tokens": 1000,
        "streaming": False
    }
    
    return model_class(**model_params)

def construct_chat_prompt(query: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
    """Construct a chat prompt with system message and context."""
    messages = []
    
    # Add system message with context if available
    system_content = "You are a helpful AI assistant that provides accurate and informative responses."
    if context:
        system_content += f"\n\nContext information:\n{context}"
    messages.append({"role": "system", "content": system_content})
    
    # Add user query
    messages.append({"role": "user", "content": query})
    
    return messages

async def get_llm_response(
    query: str,
    context: Optional[str] = None,
    provider: str = DEFAULT_MODEL_PROVIDER,
    model_name: str = DEFAULT_MODEL_NAME,
    api_key: Optional[str] = None
) -> str:
    """Get a response from the LLM."""
    try:
        # Initialize model
        model = initialize_model(provider, model_name, api_key)
        langchain_model = model.to_langchain()
        
        # Construct prompt
        messages = construct_chat_prompt(query, context)
        
        # Convert messages to LangChain format
        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            else:
                lc_messages.append(HumanMessage(content=msg["content"]))
        
        # Get response
        response = await langchain_model.ainvoke(lc_messages)
        return response.content
        
    except Exception as e:
        print(f"Error getting LLM response: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting response from language model: {str(e)}"
        )

async def fetch_context_content(context: Optional[ChatContext], db: AsyncSurreal) -> str:
    """Helper function to fetch content based on the provided context."""
    if not context:
        return "" # No specific context provided

    content_parts = []
    # Fetch source content
    if context.source_ids:
        for source_id in context.source_ids:
            if ":" not in source_id:
                print(f"Skipping invalid source ID format: {source_id}")
                continue
            try:
                # Use .get() for safer dictionary access
                source_data: Optional[dict] = await db.select(source_id)
                if source_data and source_data.get("content"):
                    title = source_data.get("title", source_id)
                    content = source_data.get("content", "")
                    content_parts.append(f"--- Source: {title} ---\n{content}\n------")
            except Exception as e:
                print(f"Error fetching source {source_id} for context: {e}")

    # Fetch note content
    if context.note_ids:
        for note_id in context.note_ids:
            if ":" not in note_id:
                print(f"Skipping invalid note ID format: {note_id}")
                continue
            try:
                # Use .get() for safer dictionary access
                note_data: Optional[dict] = await db.select(note_id)
                if note_data and note_data.get("content"):
                    title = note_data.get("title", note_id)
                    content = note_data.get("content", "")
                    content_parts.append(f"--- Note: {title} ---\n{content}\n------")
            except Exception as e:
                print(f"Error fetching note {note_id} for context: {e}")

    # TODO: Implement other context modes like "summary", "full" (fetch all sources/notes for notebook)

    return "\n\n".join(content_parts)

async def generate_citations(response_text: str, context_sources: List[Source], context_notes: List[Note]) -> List[Citation]:
    """Placeholder function to generate citations based on response and context."""
    # In a real implementation, this would involve more sophisticated logic:
    # - Identifying sentences/claims in the response.
    # - Searching for similar sentences/claims in the context_sources/context_notes content.
    # - Creating Citation objects linking response parts to specific source/note IDs.
    citations = []
    # Example: If response mentions something found in a source
    if context_sources and "placeholder text" in response_text.lower():
         # Ensure context_sources[0] is a dict or has an 'id' attribute if it's an object
         source_id_for_citation = context_sources[0].get("id") if isinstance(context_sources[0], dict) else getattr(context_sources[0], "id", "unknown_source")
         citations.append(Citation(source_id=source_id_for_citation, text="...cited snippet from source..."))
    if context_notes and "another placeholder" in response_text.lower():
         # Ensure context_notes[0] is a dict or has an 'id' attribute if it's an object
         note_id_for_citation = context_notes[0].get("id") if isinstance(context_notes[0], dict) else getattr(context_notes[0], "id", "unknown_note")
         # Note: Citation model uses source_id, might need adjustment or a different citation model for notes
         citations.append(Citation(source_id=note_id_for_citation, text="...cited snippet from note..."))

    return citations

@router.post("/sources/{source_id}/transformations", response_model=SourceInsight, status_code=status.HTTP_201_CREATED)
async def apply_transformation(
    source_id: str,
    request_data: ApplyTransformationRequest,
    x_provider_api_key: Optional[str] = Header(None, description="API Key for the AI provider (if required by the selected model)"),
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Applies a transformation to a source (e.g., summarization, key points extraction)."""
    if ":" not in source_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid source ID format.")
    try:
        # 1. Fetch the source
        source_result = await db.select(source_id)
        if not source_result:
            raise HTTPException(status_code=404, detail=f"Source {source_id} not found")
        source = source_result if isinstance(source_result, dict) else dict(source_result)
        full_text = source.get("full_text") or source.get("content")
        if not full_text:
            raise HTTPException(status_code=400, detail="Source has no content to transform.")

        # 2. Fetch the transformation
        transformation_id = request_data.transformation_id
        transformation_result = await db.select(transformation_id)
        if not transformation_result:
            raise HTTPException(status_code=404, detail=f"Transformation {transformation_id} not found")
        transformation = transformation_result if isinstance(transformation_result, dict) else dict(transformation_result)
        transformation_prompt = transformation.get("prompt")
        transformation_title = transformation.get("title", "Insight")
        if not transformation_prompt:
            raise HTTPException(status_code=400, detail="Transformation has no prompt.")

        # 3. Run the transformation prompt using the selected model
        provider = request_data.metadata.get("provider") or DEFAULT_MODEL_PROVIDER
        model_name = request_data.model_id or DEFAULT_MODEL_NAME
        # Compose the prompt
        prompt = transformation_prompt + "\n\n# INPUT\n" + full_text
        # Use the LLM
        model = initialize_model(provider, model_name, x_provider_api_key)
        langchain_model = model.to_langchain()
        lc_messages = [SystemMessage(content=transformation_prompt), HumanMessage(content=full_text)]
        response = await langchain_model.ainvoke(lc_messages)
        insight_content = response.content if hasattr(response, "content") else str(response)

        # 4. Create a new source_insight record in SurrealDB
        now = datetime.utcnow()
        insight_data = {
            "source": source_id,
            "insight_type": transformation_title,
            "content": insight_content,
            "created": now,
            "transformation_id": transformation_id,
            "metadata": {"model": model_name, "provider": provider}
        }
        created = await db.create("source_insight", insight_data)
        if not created or not isinstance(created, list) or not created[0]:
            raise HTTPException(status_code=500, detail="Failed to create source insight.")
        created_insight = created[0]
        # 5. Return the created insight as SourceInsight
        return SourceInsight(
            id=str(created_insight.get("id", "")),
            title=transformation_title,
            content=insight_content,
            transformation_id=transformation_id,
            source_id=source_id,
            created=now,
            metadata=created_insight.get("metadata", {})
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error applying transformation to source {source_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error applying transformation: {e}")

@router.post("/notebooks/by-name/{name}/chat", response_model=ChatResponse)
async def chat_with_notebook_by_name(
    name: str,
    request_data: ChatRequest,
    x_provider_api_key: Optional[str] = Header(None, description="API Key for the AI provider (if required by the selected model)"),
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Handles chat interactions within the context of a specific notebook by name."""
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

        # Fetch context content based on request_data.context
        context_str = await fetch_context_content(request_data.context, db)
        
        # Get model configuration from request or use defaults
        provider = request_data.model_provider or DEFAULT_MODEL_PROVIDER
        model_name = request_data.model_name or DEFAULT_MODEL_NAME
        
        # Get response from LLM
        answer = await get_llm_response(
            query=request_data.query,
            context=context_str,
            provider=provider,
            model_name=model_name,
            api_key=x_provider_api_key
        )
        
        # Generate citations based on response and context
        citations = await generate_citations(
            answer,
            [Source(**s) for s in request_data.context.source_ids] if request_data.context and request_data.context.source_ids else [],
            [Note(**n) for n in request_data.context.note_ids] if request_data.context and request_data.context.note_ids else []
        )

        return ChatResponse(
            answer=answer,
            citations=citations
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in chat with notebook {name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error in chat: {e}")

@router.post("/notebooks/{notebook_id}/chat", response_model=ChatResponse)
async def chat_with_notebook(
    notebook_id: str,
    request_data: ChatRequest,
    x_provider_api_key: Optional[str] = Header(None, description="API Key for the AI provider (if required by the selected model)"),
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Handles chat interactions within the context of a specific notebook."""
    if ":" not in notebook_id:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid notebook ID format.")

    try:
        # Fetch context content based on request_data.context
        context_str = await fetch_context_content(request_data.context, db)
        
        # Get model configuration from request or use defaults
        provider = request_data.model_provider or DEFAULT_MODEL_PROVIDER
        model_name = request_data.model_name or DEFAULT_MODEL_NAME
        
        # Get response from LLM
        answer = await get_llm_response(
            query=request_data.query,
            context=context_str,
            provider=provider,
            model_name=model_name,
            api_key=x_provider_api_key
        )
        
        # Generate citations based on response and context
        citations = await generate_citations(
            answer,
            [Source(**s) for s in request_data.context.source_ids] if request_data.context and request_data.context.source_ids else [],
            [Note(**n) for n in request_data.context.note_ids] if request_data.context and request_data.context.note_ids else []
        )

        return ChatResponse(
            answer=answer,
            citations=citations
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in chat with notebook {notebook_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error in chat: {e}")

@router.post("/ask", response_model=AskResponse)
async def ask_knowledge_base(
    request_data: AskRequest,
    x_provider_api_key: Optional[str] = Header(None, description="API Key for the primary AI provider (e.g., for final answer generation)"),
    db: AsyncSurreal = Depends(get_db_connection)
):
    """Answers a question based on the entire knowledge base using RAG."""
    try:
        # TODO: Implement proper RAG retrieval
        # For now, we'll just use the LLM directly with a basic prompt
        
        # Get model configuration from request or use defaults
        provider = request_data.model_provider or DEFAULT_MODEL_PROVIDER
        model_name = request_data.model_name or DEFAULT_MODEL_NAME
        
        # Get response from LLM
        answer = await get_llm_response(
            query=request_data.question,
            provider=provider,
            model_name=model_name,
            api_key=x_provider_api_key
        )
        
        # For now, return a placeholder citation
        # TODO: Implement proper citation generation based on retrieved context
        citations = [Citation(source_id="source:placeholder_rag", text="...relevant snippet from knowledge base...")]

        return AskResponse(
            answer=answer,
            citations=citations
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in ask endpoint: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error in ask: {e}")

class TransformationModel(BaseModel):
    id: Optional[str] = None
    name: str
    title: str
    description: str
    prompt: str
    apply_default: bool = False
    created: Optional[datetime] = None
    updated: Optional[datetime] = None

@router.get("/transformations", response_model=List[TransformationModel])
async def list_transformations(db: AsyncSurreal = Depends(get_db_connection)):
    """List all transformations."""
    query = f"SELECT * FROM {TRANSFORMATION_TABLE} ORDER BY updated DESC"
    result = await db.query(query)
    if not result:
        return []
    # SurrealDB may return a list of dicts or objects
    return [TransformationModel(**(r.model_dump() if hasattr(r, 'model_dump') else dict(r))) for r in result]

@router.get("/transformations/{transformation_id}", response_model=TransformationModel)
async def get_transformation(transformation_id: str, db: AsyncSurreal = Depends(get_db_connection)):
    """Get a specific transformation by ID."""
    if ":" not in transformation_id:
        raise HTTPException(status_code=400, detail="Invalid transformation ID format. Expected table:id")
    result = await db.select(transformation_id)
    if not result:
        raise HTTPException(status_code=404, detail="Transformation not found")
    return TransformationModel(**(result.model_dump() if hasattr(result, 'model_dump') else dict(result)))

@router.post("/transformations", response_model=TransformationModel, status_code=status.HTTP_201_CREATED)
async def create_transformation(transformation: TransformationModel, db: AsyncSurreal = Depends(get_db_connection)):
    """Create a new transformation."""
    now = datetime.utcnow()
    data = transformation.model_dump(exclude_unset=True)
    data["created"] = now
    data["updated"] = now
    created = await db.create(TRANSFORMATION_TABLE, data)
    if not created or not isinstance(created, list) or not created[0]:
        raise HTTPException(status_code=500, detail="Failed to create transformation")
    return TransformationModel(**(created[0].model_dump() if hasattr(created[0], 'model_dump') else dict(created[0])))

@router.patch("/transformations/{transformation_id}", response_model=TransformationModel)
async def update_transformation(transformation_id: str, transformation: TransformationModel, db: AsyncSurreal = Depends(get_db_connection)):
    """Update an existing transformation."""
    if ":" not in transformation_id:
        raise HTTPException(status_code=400, detail="Invalid transformation ID format. Expected table:id")
    update_data = transformation.model_dump(exclude_unset=True)
    update_data["updated"] = datetime.utcnow()
    updated = await db.merge(transformation_id, update_data)
    if not updated:
        raise HTTPException(status_code=404, detail="Transformation not found")
    return TransformationModel(**(updated.model_dump() if hasattr(updated, 'model_dump') else dict(updated)))

@router.delete("/transformations/{transformation_id}", response_model=StatusResponse)
async def delete_transformation(transformation_id: str, db: AsyncSurreal = Depends(get_db_connection)):
    """Delete a transformation."""
    if ":" not in transformation_id:
        raise HTTPException(status_code=400, detail="Invalid transformation ID format. Expected table:id")
    deleted = await db.delete(transformation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Transformation not found or already deleted")
    return StatusResponse(status="success", message=f"Transformation {transformation_id} deleted successfully")

