from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .services.document_processor import DocumentProcessor
from .services.retrieval import RetrievalService
from .services.analysis import AnalysisService
from .core.config import Settings
from .models.schemas import AnalysisResponse
import logging

app = FastAPI(title="CV Analyzer API")
settings = Settings()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
doc_processor = DocumentProcessor()
retrieval_service = RetrievalService()
analysis_service = AnalysisService()

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_cv(file: UploadFile = File(...)):
    """
    Analyze a CV/resume file and provide detailed feedback
    """
    try:
        # Extract text from document
        text = await doc_processor.process_document(file)
        
        # Perform retrieval and analysis
        relevant_context = retrieval_service.retrieve(text)
        analysis_result = analysis_service.analyze(text, relevant_context)
        
        return analysis_result
    except Exception as e:
        logging.error(f"Error analyzing CV: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}