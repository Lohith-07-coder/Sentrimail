"""Schemas for complaint submissions and complaint-related API requests."""

from typing import Annotated, Optional

from fastapi import Form
from pydantic import BaseModel, ConfigDict


class ComplaintCreateRequest(BaseModel):
    """Fields accepted by the complaint submission form."""

    model_config = ConfigDict(extra="ignore")

    title: str
    description: str
    language: Optional[str] = None
    # Bug #2 fix (MODEL_COMPARISON.md §4.1): capture the user-selected category
    # so it is forwarded to analyze_complaint() instead of being hardcoded to "other".
    category: str = "other"

    @classmethod
    def as_form(
        cls,
        title: Annotated[str, Form()],
        description: Annotated[str, Form()],
        language: Annotated[Optional[str], Form()] = None,
        category: Annotated[str, Form()] = "other",
    ) -> "ComplaintCreateRequest":
        """Build the schema from the established HTML form fields."""
        return cls(title=title, description=description, language=language, category=category)


class ComplaintTrackingRequest(BaseModel):
    """Complaint code submitted by the public tracking form."""

    model_config = ConfigDict(extra="ignore")

    complaint_code: str

    @classmethod
    def as_form(
        cls, complaint_code: Annotated[str, Form()]
    ) -> "ComplaintTrackingRequest":
        """Build the schema from the established tracking form field."""
        return cls(complaint_code=complaint_code)


class ComplaintAnalysisRequest(BaseModel):
    """JSON payload accepted by the existing AI analysis endpoint."""

    model_config = ConfigDict(extra="ignore")

    text: str = ""
