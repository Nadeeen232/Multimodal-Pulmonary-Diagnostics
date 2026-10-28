"""Explicit API contracts for proxy labels and uncalibrated scores."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model_version: str | None


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    probability: float = Field(
        ge=0,
        le=1,
        description="Uncalibrated score for the weak abnormality label; not a clinical risk estimate",
    )
    predicted_label: Literal["proxy_abnormal", "proxy_negative"]
    threshold: float = Field(ge=0, le=1)
    priority_tier: Literal["higher_review", "review", "lower_review"]
    priority_reason: str
    model_version: str
    calibrated: Literal[False]


class TextHighlight(BaseModel):
    field: Literal["indication", "comparison"]
    token: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    contribution: float


class ReportPredictionResponse(PredictionResponse):
    predicted_class: Literal["Normal", "Abnormal"]
    generated_report: str
    gradcam_region: str


class ExplanationResponse(BaseModel):
    predicted_class: Literal["Normal", "Abnormal"]
    prediction: PredictionResponse
    gradcam_base64: str
    text_highlights: list[TextHighlight]
    method: str
    text_tokens_examined: int
    text_tokens_truncated: bool
    generated_report: str
    gradcam_region: str
