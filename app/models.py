"""Pydantic models for request/response validation (kept loose for ProtoJSON)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Part(BaseModel):
    mediaType: Optional[str] = None
    data: Optional[Any] = None
    text: Optional[str] = None


class Message(BaseModel):
    messageId: str
    role: str
    taskId: Optional[str] = None
    contextId: Optional[str] = None
    parts: List[Part] = Field(default_factory=list)


class Configuration(BaseModel):
    returnImmediately: Optional[bool] = False
    historyLength: Optional[int] = 20
    acceptedOutputModes: Optional[List[str]] = None


class SendMessageRequest(BaseModel):
    message: Message
    configuration: Optional[Configuration] = None


class Facts(BaseModel):
    vendorName: str
    invoiceNumber: str
    amountMinor: int
    currency: str = "INR"


class Proposal(BaseModel):
    packageId: str
    actionId: str
    action: str
    facts: Facts
    evidenceRefs: List[str]
    rationale: str


class ProposalsArtifact(BaseModel):
    batchId: str
    proposals: List[Proposal]


class ResultItem(BaseModel):
    packageId: str
    actionId: str
    action: str
    outcome: str  # ACCEPTED | REJECTED
    receiptNonce: Optional[str] = None


class ResultsArtifact(BaseModel):
    batchId: str
    results: List[ResultItem]


class Execution(BaseModel):
    packageId: str
    actionId: str
    action: str
    receiptNonce: str
    facts: Facts
    evidenceRefs: List[str]


class ReceiptsArtifact(BaseModel):
    batchId: str
    executions: List[Execution]
