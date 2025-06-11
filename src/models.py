# /home/ubuntu/open_notebook_full_backend/fastapi_backend/src/models.py

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

# --- Common Models ---

class StatusResponse(BaseModel):
    status: str = "success"
    message: Optional[str] = None

class TaskStatus(BaseModel):
    task_id: str
    status: str # e.g., "pending", "running", "completed", "failed"
    message: Optional[str] = None
    result: Optional[dict] = None # Could hold result upon completion

# --- Notebook Models ---

class NotebookBase(BaseModel):
    name: str = Field(..., example="My Research Project")
    description: Optional[str] = Field(None, example="Notes and sources related to quantum entanglement.")

class NotebookCreate(NotebookBase):
    pass

class NotebookUpdate(BaseModel):
    name: Optional[str] = Field(None, example="Updated Project Name")
    description: Optional[str] = Field(None, example="Updated description with more details.")

# Summary model for lists
class NotebookSummary(NotebookBase):
    id: str
    created: datetime
    updated: datetime
    archived: bool = False
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)  # For storing notebook-specific metadata

# Full notebook model with relationships
class Notebook(NotebookSummary):
    sources: List["SourceSummary"] = Field(default_factory=list)  # Related sources
    notes: List["NoteSummary"] = Field(default_factory=list)      # Related notes
    chat_sessions: List["ChatSessionSummary"] = Field(default_factory=list)  # Related chat sessions

# --- Note Models ---

class NoteBase(BaseModel):
    title: str
    content: str

class NoteCreate(NoteBase):
    """Simplified model for creating notes with just title and content."""
    pass

class NoteUpdate(NoteBase):
    """Model for updating notes with optional title and content."""
    title: Optional[str] = None
    content: Optional[str] = None

# Summary model for lists within a notebook
class NoteSummary(BaseModel):
    id: str
    title: str
    created: datetime
    updated: datetime
    note_type: str = "human"

# Full note model
class Note(NoteSummary):
    content: str
    notebook_id: Optional[str] = None

# Response model for notes
class NoteResponse(BaseModel):
    id: str
    title: str
    content: str
    created: datetime
    updated: datetime
    note_type: str = "human"
    notebook_id: Optional[str] = None

# --- Source Models ---

class SourceSummary(BaseModel):
    id: str
    title: str
    type: str  # e.g., "url", "pdf", "youtube"
    status: str  # e.g., "pending", "processing", "completed", "failed"
    created: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Source(SourceSummary):
    content: Optional[str] = None  # Extracted text content
    notebook_id: Optional[str] = None
    embedding: List[float] = Field(default_factory=list)  # For vector search
    needs_embedding: bool = Field(default=True)  # Flag for embedding generation

# --- Chat Session Models ---

class ChatSessionSummary(BaseModel):
    id: str
    title: Optional[str] = None
    created: datetime
    updated: datetime
    notebook_id: Optional[str] = None

class ChatSession(ChatSessionSummary):
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# --- AI Interaction Models ---

class SourceInsight(BaseModel):
    id: str
    title: str
    content: str
    transformation_id: str
    source_id: str
    created: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Citation(BaseModel):
    source_id: str
    text: str  # The cited text snippet
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatContext(BaseModel):
    source_ids: Optional[List[str]] = None
    note_ids: Optional[List[str]] = None
    mode: Literal["summary", "full", "auto", "list_of_ids"] = "auto"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ApplyTransformationRequest(BaseModel):
    transformation_id: str
    model_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(protected_namespaces=())

class ChatRequest(BaseModel):
    query: str
    context: Optional[ChatContext] = None
    model_provider: Optional[str] = Field(None, description="The provider of the language model (e.g., 'openai', 'anthropic', 'groq')")
    model_name: Optional[str] = Field(None, description="The name of the language model to use (e.g., 'gpt-3.5-turbo', 'claude-3-opus')")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(protected_namespaces=())

class AskRequest(BaseModel):
    question: str
    model_provider: Optional[str] = Field(None, description="The provider of the language model (e.g., 'openai', 'anthropic', 'groq')")
    model_name: Optional[str] = Field(None, description="The name of the language model to use (e.g., 'gpt-3.5-turbo', 'claude-3-opus')")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(protected_namespaces=())

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SourceResponse(BaseModel):
    id: str
    title: Optional[str] = None
    content: Optional[str] = None
    created: datetime
    updated: datetime
    metadata: Optional[Dict[str, Any]] = None

class NotebookResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    created: datetime
    updated: datetime
    metadata: Optional[Dict[str, Any]] = None

class NotebookWithNotesResponse(NotebookSummary):
    notes: List[NoteResponse] = Field(default_factory=list)
    sources: List[SourceSummary] = Field(default_factory=list)
    chat_sessions: List[ChatSessionSummary] = Field(default_factory=list)

class SourceInsightResponse(BaseModel):
    id: str
    insight_type: str
    content: str
    created: datetime
    metadata: Dict[str, Any] = {}

class SourceWithInsightsResponse(BaseModel):
    id: str
    title: str
    insights: List[SourceInsightResponse] = []

class NoteFullResponse(BaseModel):
    id: str
    title: str
    content: str
    created: datetime
    updated: datetime
    note_type: str
    metadata: Dict[str, Any]
    notebook_id: Optional[str]
    source: Optional[SourceWithInsightsResponse] = None

# --- Model Configuration Models ---
class ModelBase(BaseModel):
    name: str
    provider: str
    type: Literal["language", "embedding", "text_to_speech", "speech_to_text"]
    model_config = ConfigDict(from_attributes=True)

class ModelCreate(ModelBase):
    pass

class ModelUpdate(BaseModel):
    name: Optional[str] = None
    provider: Optional[str] = None
    type: Optional[Literal["language", "embedding", "text_to_speech", "speech_to_text"]] = None
    model_config = ConfigDict(from_attributes=True)

class Model(ModelBase):
    id: str
    created: datetime
    updated: datetime

class DefaultModels(BaseModel):
    """API model for default model configurations.
    This matches the domain model structure but uses Pydantic v2 features.
    """
    # Model configuration fields
    default_chat_model: Optional[str] = None
    default_transformation_model: Optional[str] = None
    large_context_model: Optional[str] = None
    default_text_to_speech_model: Optional[str] = None
    default_speech_to_text_model: Optional[str] = None
    default_embedding_model: Optional[str] = None
    default_tools_model: Optional[str] = None
    
    # Database metadata fields
    id: Optional[str] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
    
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        extra='ignore',
        str_strip_whitespace=True,
        validate_assignment=True
    )

# Add other models as needed for Podcast, Search, Config etc.

