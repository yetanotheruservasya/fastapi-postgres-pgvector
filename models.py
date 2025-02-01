"""
Module containing data models used across the project.
"""

from pydantic import BaseModel

# Data model for company information
class CompanyData(BaseModel):
    """
    Represents the company data structure with source, company_id, and additional data.
    """
    source: str
    company_id: str
    data: dict
