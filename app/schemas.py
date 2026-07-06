from datetime import datetime

from pydantic import BaseModel, Field


class BankClientSummary(BaseModel):
    client_id: str
    name: str
    risk_tolerance: str


class BankClientDetail(BaseModel):
    client_id: str
    profile_created_at: str
    name: str
    age: int
    income_type: str
    monthly_income_myr: float
    risk_tolerance: str
    dependents: int
    employment: str
    existing_products: list[dict]
    goals: list[str]
    notes: str
    history_seed: list[str]


class SessionResponse(BaseModel):
    id: str
    client_id: str
    created_at: datetime


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: datetime


class ChatResponse(BaseModel):
    session_id: str
    client_id: str
    reply: str
    messages: list[ChatMessage]


class DeleteClientDataResponse(BaseModel):
    client_id: str
    sessions_deleted: int
    messages_deleted: int
from datetime import datetime

from pydantic import BaseModel, Field


class ClientSummary(BaseModel):
    client_id: str
    name: str
    risk_tolerance: str


class ClientDetail(BaseModel):
    client_id: str
    profile_created_at: str
    name: str
    age: int
    income_type: str
    monthly_income_myr: float
    risk_tolerance: str
    dependents: int
    employment: str
    existing_products: list[dict]
    goals: list[str]
    notes: str
    history_seed: list[str]


class SessionResponse(BaseModel):
    id: str
    client_id: str
    created_at: datetime


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: datetime


class ChatResponse(BaseModel):
    session_id: str
    client_id: str
    reply: str
    messages: list[ChatMessage]


class DeleteClientDataResponse(BaseModel):
    client_id: str
    sessions_deleted: int
    messages_deleted: int
