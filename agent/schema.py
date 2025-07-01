from pydantic import BaseModel, Field
from typing import List, Optional

class ResearchReport(BaseModel):
    executive_summary: str = Field(..., description="A brief overview of the research objective and key outcomes.")
    key_findings: List[str] = Field(..., description="Concise, cited insights from the research.")
    visuals: Optional[List[str]] = Field(None, description="A list of file IDs for any generated charts or images.")
    conclusion: str = Field(..., description="A summary of key takeaways and suggested next steps.")
    references: List[str] = Field(..., description="A list of sources used in the research.")

class ThinkingProcess(BaseModel):
    reasoning_steps: List[str] = Field(..., description="A breakdown of the agent's reasoning steps.")
    tools_used: List[str] = Field(..., description="A list of tools and sources used during the research.")
    decisions_made: List[str] = Field(..., description="Key decisions made during planning and execution.")
