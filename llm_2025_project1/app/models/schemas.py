from pydantic import BaseModel
from typing import List, Dict, Optional, Union
from enum import Enum

class SkillLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class LanguageLevel(str, Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    FLUENT = "fluent"
    NATIVE = "native"

class SkillType(str, Enum):
    TECHNICAL = "technical"
    LANGUAGE = "language"
    PROFESSIONAL = "professional"

class Skill(BaseModel):
    name: str
    level: Union[SkillLevel, LanguageLevel]
    type: SkillType
    context: Optional[str] = None

class Experience(BaseModel):
    company: str
    position: str
    duration: str
    highlights: List[str]
    impact_score: float

class Education(BaseModel):
    institution: str
    degree: str
    field: str
    year: str

class ImprovementSuggestion(BaseModel):
    section: str
    current_state: str
    suggestion: str
    priority: int  # 1-5, with 5 being highest priority

class CVAnalysis(BaseModel):
    skills: List[Skill]
    experience: List[Experience]
    education: List[Education]
    overall_score: float  # 0-100
    improvement_suggestions: List[ImprovementSuggestion]
    strengths: List[str]
    weaknesses: List[str]
    industry_fit: List[str]
    keyword_optimization: Dict[str, float]

class AnalysisResponse(BaseModel):
    success: bool
    analysis: Optional[CVAnalysis] = None
    error: Optional[str] = None