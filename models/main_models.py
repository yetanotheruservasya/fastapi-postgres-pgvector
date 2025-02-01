"""
Module containing data models used across the project.
"""
from typing import List, Optional
from pydantic import BaseModel, Field

class EntityData(BaseModel):
    """
    Represents the company data structure with source, company_id, and additional data.
    """
    source: str
    entity_id: str  # ранее было company_id
    data: dict

class EntityNormalizedData(BaseModel):
    """
    Represents a normalized entity with specific fields and an optional search vector.
    """
    entity_id: str = Field(..., description="Entity identifier")
    name: str = Field(..., description="Entity name")
    industry: Optional[str] = Field(None, description="Industry")
    description: Optional[str] = Field(None, description="Entity description")
    vector: Optional[List[float]] = Field(None, description="Vector for search")
