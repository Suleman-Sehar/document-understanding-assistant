from pydantic import BaseModel, Field, field_validator
from typing import List

class DocResult(BaseModel):
    doc_type: str
    fields: dict
    summary: str
    confidence: float
    needs_human_review: bool
    issues: List[str]=Field(default_factory=list)

    @field_validator("fields")
    @classmethod
    def fields_nonempty(cls,v):
        if not v:
            raise ValueError("fields must not be empty")
        return v

