from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.output_parsers import PydanticOutputParser
from ..models.schemas import CVAnalysis, AnalysisResponse
from ..core.config import Settings

class AnalysisService:
    """
    Handles CV analysis using LLM and structured output parsing
    """
    
    def __init__(self):
        self.settings = Settings()
        self.llm = ChatOpenAI(
            api_key=self.settings.OPENAI_API_KEY,
            model=self.settings.OPENAI_MODEL_NAME,
            temperature=self.settings.TEMPERATURE,
            max_tokens=self.settings.MAX_TOKENS
        )
        self.output_parser = PydanticOutputParser(pydantic_object=CVAnalysis)
        
    def analyze(self, cv_text: str, relevant_context: List[Dict]) -> AnalysisResponse:
        """
        Analyze CV text and generate structured feedback
        """
        try:
            # Create analysis prompt
            prompt = self._create_analysis_prompt(cv_text, relevant_context)
            
            # Get LLM response
            response = self.llm.invoke(prompt)
            
            # Extract content from AIMessage
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Parse response into structured format
            analysis = self.output_parser.parse(response_text)
            
            return AnalysisResponse(
                success=True,
                analysis=analysis
            )
            
        except Exception as e:
            return AnalysisResponse(
                success=False,
                error=str(e)
            )
            
    def _create_analysis_prompt(self, cv_text: str, relevant_context: List[Dict]) -> List:
        """
        Create a detailed prompt for CV analysis
        """
        system_message = SystemMessage(content="""You are an expert CV analyzer. Your task is to analyze CVs and provide detailed, structured feedback.
        You must format your response as a valid JSON object matching the provided schema.
        Be specific, actionable, and professional in your analysis.""")
        
        context_str = "\n".join([item["content"] for item in relevant_context])
        
        human_template = """Please analyze the following CV and provide detailed feedback.
        Use the provided context for industry standards and best practices.
        
        CV Text:
        {cv_text}
        
        Relevant Context:
        {context}
        
        Provide a detailed analysis including:
        1. Skills assessment:
           - For technical skills: use levels (beginner, intermediate, advanced, expert)
           - For languages: use levels (basic, intermediate, advanced, fluent, native)
           - For professional skills: use levels (beginner, intermediate, advanced, expert)
           - Mark each skill with appropriate type: "technical", "language", or "professional"
        2. Experience evaluation with impact scores
        3. Education details
        4. Overall score (0-100)
        5. Specific improvement suggestions
        6. Key strengths and weaknesses
        7. Industry fit analysis
        8. Keyword optimization recommendations
        
        Your response MUST be a valid JSON object matching this schema:
        {format_instructions}
        """
        
        human_message = HumanMessage(content=human_template.format(
            cv_text=cv_text,
            context=context_str,
            format_instructions=self.output_parser.get_format_instructions()
        ))
        
        return [system_message, human_message]