from dataclasses import dataclass, field
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Text = Annotated[str, StringConstraints(strict=True, min_length=1, pattern=r"\S")]
Scope = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=512, pattern=r"\S")]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Message(ContractModel):
    role: Literal["user", "assistant"]
    content: Text
    timestamp: Annotated[int, Field(strict=True, ge=0)] | None = None


class AddRequest(ContractModel):
    request_id: Scope
    messages: Annotated[list[Message], Field(min_length=1)]
    user_id: Scope
    session_id: Scope


class AddReceipt(ContractModel):
    success: Literal[True] = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(ContractModel):
    query: Text
    options: Annotated[list[Text], Field(min_length=2)] | None = None
    user_id: Scope
    top_k: Annotated[int, Field(strict=True, ge=1, le=100)]


@dataclass(frozen=True)
class Source:
    id: str
    user_id: str
    session_id: str
    request_id: str
    position: int
    role: str
    content: str
    timestamp: int | None = None


@dataclass(frozen=True)
class Evidence:
    id: str
    content: str
    source_ids: tuple[str, ...]
    score: float = 0.0
    kind: str = "source"


@dataclass(frozen=True)
class Fact:
    text: str
    source_ids: tuple[str, ...]
    subject: str
    predicate: str
    value: str
    effective_time: str
    attribution: Literal["user", "assistant", "unknown"]
    status: Literal["current", "historical", "negated", "uncertain"]
    supersedes: tuple[str, ...] = ()


class RequestConflict(ValueError):
    pass


class Memory(Protocol):
    def add(self, request: AddRequest) -> AddReceipt: ...
    def search(self, request: SearchRequest) -> list[Evidence]: ...


class Extractor(Protocol):
    @property
    def strategy_identity(self) -> dict[str, object]: ...

    def extract(self, sources: list[Source]) -> list[Fact]: ...


@dataclass(frozen=True)
class ModelConfig:
    model: str
    max_tokens: int
    effort: str | None = None
    temperature: float | None = None
    timeout_s: float = 180.0
    retries: int = 2


@dataclass(frozen=True)
class Completion:
    text: str
    stop_reason: str
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    attempt_id: str = ""


class Provider(Protocol):
    def complete(self, *, config: ModelConfig, prompt: str, system: str = "") -> Completion: ...
