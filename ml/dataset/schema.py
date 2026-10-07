"""
SentriMail Unified Multi-Task Dataset Schema
--------------------------------------------
Defines the standard data record structure across all dataset sources.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class UnifiedSample(BaseModel):
    """
    Standardized multi-task dataset sample containing text, tags, and targets.
    """
    id: str = Field(..., description="Unique sample identifier")
    text: str = Field(..., description="Raw or cleaned customer complaint/query text")
    language: str = Field(default="en", description="ISO 639-1 language code")
    
    # Classification Head Targets
    message_type: str = Field(default="complaint", description="complaint | query | praise | suggestion | spam_abuse")
    category: str = Field(default="other", description="billing | technical | delivery | customer_service | product | refund | other")
    issue: str = Field(default="general", description="billing | auth | technical | delivery | support | general")
    sentiment: str = Field(default="NEUTRAL", description="POSITIVE | NEUTRAL | NEGATIVE")
    emotion: str = Field(default="neutral", description="anger | fear | sadness | disgust | surprise | joy | neutral")
    
    # Seq2Seq Generation Targets
    root_cause: str = Field(default="", description="Silver or gold root cause summary string")
    response: str = Field(default="", description="Ground-truth or template resolution response")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
