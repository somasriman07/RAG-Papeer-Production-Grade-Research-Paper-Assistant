from typing import Literal
from pydantic import BaseModel, Field


class BtwRouteDecision(BaseModel):
    needs_web_search: bool = False


class RouterDecision(BaseModel):
    route: Literal["retrieve", "verify_claim", "direct_answer"] = "retrieve"


class RelevancyDecision(BaseModel):
    is_relevant: bool = True
    reason: str = ""


class SupersedingPaper(BaseModel):
    title: str = ""
    url: str = ""
    summary: str = ""


class ClaimVerificationResult(BaseModel):
    is_superseded: bool = False
    verdict_summary: str = ""
    superseding_papers: list[SupersedingPaper] = Field(default_factory=list)