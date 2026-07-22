"""Schemas for user authentication requests."""

from typing import Annotated

from fastapi import Form
from pydantic import BaseModel, ConfigDict


class LoginRequest(BaseModel):
    """Credentials submitted by the login form."""

    model_config = ConfigDict(extra="ignore")

    username: str
    password: str

    @classmethod
    def as_form(
        cls,
        username: Annotated[str, Form()],
        password: Annotated[str, Form()],
    ) -> "LoginRequest":
        """Build the schema from an HTML form without changing its field names."""
        return cls(username=username, password=password)


class RegisterRequest(LoginRequest):
    """Registration fields submitted by the existing registration form."""

    email: str

    @classmethod
    def as_form(
        cls,
        username: Annotated[str, Form()],
        password: Annotated[str, Form()],
        email: Annotated[str, Form()],
    ) -> "RegisterRequest":
        """Build the schema from an HTML form without changing its field names."""
        return cls(username=username, password=password, email=email)
