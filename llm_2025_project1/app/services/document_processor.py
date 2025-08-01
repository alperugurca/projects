from fastapi import UploadFile
from pdfminer.high_level import extract_text
from docx import Document
import io
import logging

class DocumentProcessor:
    """
    Handles document processing and text extraction from various file formats
    """
    
    async def process_document(self, file: UploadFile) -> str:
        """
        Process uploaded document and extract text
        """
        try:
            content = await file.read()
            
            if file.filename.endswith('.pdf'):
                return self._process_pdf(content)
            elif file.filename.endswith('.docx'):
                return self._process_docx(content)
            else:
                raise ValueError("Unsupported file format. Please upload PDF or DOCX files only.")
                
        except Exception as e:
            logging.error(f"Error processing document: {str(e)}")
            raise
            
    def _process_pdf(self, content: bytes) -> str:
        """
        Extract text from PDF file
        """
        try:
            pdf_file = io.BytesIO(content)
            text = extract_text(pdf_file)
            return self._clean_text(text)
        except Exception as e:
            logging.error(f"Error processing PDF: {str(e)}")
            raise
            
    def _process_docx(self, content: bytes) -> str:
        """
        Extract text from DOCX file
        """
        try:
            docx_file = io.BytesIO(content)
            doc = Document(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return self._clean_text(text)
        except Exception as e:
            logging.error(f"Error processing DOCX: {str(e)}")
            raise
            
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        """
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove special characters that might interfere with analysis
        text = text.replace("\x0c", " ")
        
        return text.strip()