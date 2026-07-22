"""Schemas for administrative complaint actions."""

from typing import Annotated

from fastapi import Form
from pydantic import BaseModel, ConfigDict


class ComplaintStatusUpdateRequest(BaseModel):
    """Status update submitted by an administrator."""

    model_config = ConfigDict(extra="ignore")

    status: str

    @classmethod
    def as_form(
        cls, status: Annotated[str, Form()]
    ) -> "ComplaintStatusUpdateRequest":
        """Build the schema from the established administration form field."""
        return cls(status=status)


class ComplaintResponseUpdateRequest(BaseModel):
    """Administrator response and its desired complaint status."""

    model_config = ConfigDict(extra="ignore")

    response: str
    status: str = "resolved"

    @classmethod
    def as_form(
        cls,
        response: Annotated[str, Form()],
        status: Annotated[str, Form()] = "resolved",
    ) -> "ComplaintResponseUpdateRequest":
        """Build the schema from the established administration form fields."""
        return cls(response=response, status=status)
