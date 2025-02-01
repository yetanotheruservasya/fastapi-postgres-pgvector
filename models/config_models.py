"""
Module containing configuration models for data processing.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field

class FieldConfig(BaseModel):
    """
    Configuration for a specific field.
    
    Attributes:
        source_field: JSON path from which to extract the value.
        required: Indicates whether the field is required.
        vectorize: Specifies whether this field should be used for vectorization.
    """
    source_field: str = Field(
        ...,
        description="JSON path from which to extract the value"
    )
    required: bool = Field(
        False,
        description="Indicates whether the field is required"
    )
    vectorize: Optional[bool] = Field(
        False,
        description="Specifies whether this field should be used for vectorization"
    )

class VectorSettings(BaseModel):
    """
    Settings for vectorization.
    
    Attributes:
        vector_field: Name of the normalized field used for vectorization.
        model: Model used for generating the vector.
    """
    vector_field: str = Field(
        ...,
        description="Name of the normalized field used for vectorization"
    )
    model: str = Field(
        ...,
        description="Model used for generating the vector"
    )

class EntityConfig(BaseModel):
    """
    Configuration for an entity.
    
    Attributes:
        entity_name: Name of the entity.
        fields: Dictionary describing the configuration of fields.
        vector_settings: Optional settings for vectorization.
    """
    entity_name: str = Field(
        ...,
        description="Name of the entity"
    )
    fields: Dict[str, FieldConfig] = Field(
        ...,
        description="Dictionary describing the configuration of fields"
    )
    vector_settings: Optional[VectorSettings] = Field(
        None,
        description="Optional settings for vectorization"
    )
