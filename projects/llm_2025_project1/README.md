# Video:
[![Watch the video](https://i.ytimg.com/vi/1O2yPlnN2n8/maxresdefault.jpg)](https://youtu.be/1O2yPlnN2n8)


# CV Analyzer

An end-to-end RAG (Retrieval Augmented Generation) application for analyzing resumes and providing detailed feedback.

## Problem Description

The CV Analyzer is designed to help both job seekers and recruiters by providing detailed, AI-powered analysis of resumes/CVs. The application:
- Extracts text from PDF and DOCX resumes
- Analyzes key components (skills, experience, education, etc.)
- Provides actionable feedback on improvements
- Evaluates resume against industry best practices
- Generates tailored suggestions for enhancement

## Features

- Multiple document format support (PDF, DOCX)
- Hybrid search combining vector and text-based retrieval
- Document re-ranking for improved accuracy
- Query rewriting for better context understanding
- Interactive web interface
- Real-time analysis dashboard
- User feedback collection
- Automated ingestion pipeline

## Technical Architecture

1. **Frontend**: Streamlit web interface
2. **Backend**: FastAPI
3. **RAG Components**:
   - Document Processing: pdfminer.six, python-docx
   - Vector Store: ChromaDB
   - LLM Integration: LangChain + OpenAI
   - Hybrid Search: BM25 + Embedding similarity
4. **Monitoring**: Custom dashboard with Plotly

## Setup Instructions

1. Clone the repository:
```bash
git clone [repository-url]
cd cv-analyzer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
```

5. Run the application:
```bash
# Start the FastAPI backend
uvicorn app.main:app --reload

# In a new terminal, start the Streamlit frontend
streamlit run frontend/streamlit_app.py
```

## Project Structure

```
cv-analyzer/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── logging.py
│   ├── models/
│   │   └── schemas.py
│   └── services/
│       ├── document_processor.py
│       ├── retrieval.py
│       └── analysis.py
├── frontend/
│   └── streamlit_app.py
├── data/
│   └── knowledge_base/
├── scripts/
│   └── ingest.py
├── tests/
│   └── __init__.py
├── requirements.txt
└── README.md
```

## Monitoring Dashboard

The application includes a monitoring dashboard with the following metrics:
1. Analysis requests over time
2. Average processing time
3. User satisfaction ratings
4. Most common improvement areas
5. Document format distribution

## Evaluation Results

### Retrieval Approaches Evaluated
1. Pure BM25
2. Pure Vector Search
3. Hybrid (BM25 + Vector) with weights
4. Hybrid with re-ranking

### LLM Approaches Evaluated
1. Direct question-answering
2. Chain-of-thought prompting
3. Structured output with scoring
4. Multi-step analysis with feedback aggregation

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
