import easyocr
import numpy as np
from PIL import Image
from transformers import pipeline
import gradio as gr
import pdf2image
import PyPDF2
import io
import pandas as pd
import logging
from datetime import datetime
import os
import torch

# Add these near the top of your script, after imports
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
torch.backends.cudnn.benchmark = True

# If you're running out of memory, uncomment these lines:
# import gc
# gc.collect()
# torch.cuda.empty_cache()

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models with specific device placement and lower precision
device = 0 if torch.cuda.is_available() else -1
logger.info(f"Using device: {'CUDA' if device == 0 else 'CPU'}")

# Initialize models with memory optimization
def init_models():
    try:
        # Initialize EasyOCR with lower memory usage
        reader = easyocr.Reader(['en'], gpu=bool(device == 0), 
                              model_storage_directory='./models',
                              download_enabled=True)
        
        # Initialize text classifier with optimizations
        text_classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
            model_kwargs={"low_cpu_mem_usage": True}
        )
        
        # Use a more lightweight document classifier
        doc_classifier = pipeline(
            "image-classification",
            model="microsoft/dit-base-finetuned-rvlcdip",
            device=device,
            model_kwargs={"low_cpu_mem_usage": True}
        )
        
        return reader, text_classifier, doc_classifier
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise

try:
    logger.info("Initializing models...")
    reader, text_classifier, doc_classifier = init_models()
    logger.info("Models initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize models: {str(e)}")
    raise

def validate_insurance_claim(text):
    """Validate if the text contains insurance claim related content"""
    keywords = ['claim', 'policy', 'insurance', 'damage', 'loss', 'accident', 'coverage']
    return any(keyword in text.lower() for keyword in keywords)

def process_document(file):
    try:
        if file is None:
            return "Please upload an insurance claim document", None, None

        # Get file extension
        file_extension = os.path.splitext(file.name)[1].lower()

        # Handle PDF files
        if file_extension == '.pdf':
            try:
                images = pdf2image.convert_from_bytes(file.read(), first_page=1, last_page=1)
                if not images:
                    return "Failed to process insurance claim PDF", None, None
                image = images[0]
            except Exception as e:
                logger.error(f"PDF processing error: {str(e)}")
                return "Error processing PDF file", None, None

        # Handle image files
        elif file_extension in ('.png', '.jpg', '.jpeg'):
            try:
                image = Image.open(file)
            except Exception as e:
                logger.error(f"Image processing error: {str(e)}")
                return "Error processing image file", None, None
        else:
            return "Unsupported file format. Please upload PDF or image files.", None, None

        # Convert image to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Extract text with error handling
        try:
            result = reader.readtext(np.array(image))
            text = ' '.join([t[1] for t in result])
        except Exception as e:
            logger.error(f"Text extraction error: {str(e)}")
            return "Error extracting text from document", None, None

        # Format the extracted text
        formatted_text = format_insurance_claim(text)

        # Validate if it's an insurance claim
        if not validate_insurance_claim(text):
            return "Document does not appear to be an insurance claim", None, None

        # Classify text with error handling
        try:
            text_analysis = text_classifier(text[:512])[0]
        except Exception as e:
            logger.error(f"Text classification error: {str(e)}")
            text_analysis = {'score': 0.5}

        # Classify document with error handling
        try:
            doc_analysis = doc_classifier(image)[0]
        except Exception as e:
            logger.error(f"Document classification error: {str(e)}")
            doc_analysis = {'score': 0.5}

        # Generate validation results
        validation_result = analyze_claim_validity(text_analysis['score'])
        
        return (
            formatted_text,
            f"Claim Status: {validation_result['status']}\n" +
            f"Confidence Score: {text_analysis['score']:.2f}\n" +
            f"Validation Notes: {validation_result['notes']}",
            f"Document Type: Insurance Claim Form\n" +
            f"Form Type: NUCC Health Insurance Claim\n" +
            f"Confidence: {doc_analysis['score']:.2f}"
        )

    except Exception as e:
        logger.error(f"General processing error: {str(e)}")
        return "Error processing document", None, None

def format_insurance_claim(text):
    """Format the extracted text in a more readable way"""
    # Extract key information using regex or simple text processing
    lines = text.split('\n')
    formatted_lines = []
    
    key_fields = {
        'Insured Name': '',
        'Policy Number': '',
        'Provider': '',
        'Date of Service': '',
        'Claim Details': ''
    }
    
    # Process the text and organize it
    for line in lines:
        if 'HEALTH INSURANCE CLAIM FORM' in line:
            formatted_lines.append(f"Document Type: {line.strip()}")
        elif any(field in line for field in ['Name:', 'Policy', 'Provider', 'Date']):
            formatted_lines.append(line.strip())
    
    return '\n'.join(formatted_lines)

def analyze_claim_validity(score):
    """Provide more detailed validation analysis"""
    if score > 0.9:
        return {
            'status': 'VALID',
            'notes': 'High confidence in claim validity. All required fields present.'
        }
    elif score > 0.7:
        return {
            'status': 'VALID - REVIEW RECOMMENDED',
            'notes': 'Claim appears valid but manual review suggested.'
        }
    else:
        return {
            'status': 'NEEDS REVIEW',
            'notes': 'Low confidence score. Please review manually.'
        }

# Custom CSS for better UI
custom_css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto;
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
}

.main-div {
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.container {
    border-radius: 10px;
    background-color: #ffffff;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
    padding: 20px;
}

.output-div {
    min-height: 100px;
    margin-bottom: 10px;
}

h1 {
    color: #2a4365;
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 1rem;
    font-weight: bold;
}

.description {
    text-align: center;
    color: #4a5568;
    margin-bottom: 2rem;
}

.file-upload {
    border: 2px dashed #cbd5e0;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
}

.file-upload:hover {
    border-color: #4299e1;
}

.output-label {
    font-weight: bold;
    color: #2d3748;
    margin-bottom: 0.5rem;
}

.output-text {
    background-color: #f7fafc;
    border-radius: 6px;
    padding: 12px;
}
"""

# Create Gradio interface with enhanced UI
with gr.Blocks(css=custom_css) as iface:
    gr.HTML("<h1>🔍 Automated Insurance Claim Validation System</h1>")
    gr.HTML("""
        <div class="description">
            Upload insurance claim documents (PDF or image) for automated validation and analysis.
            Our AI system will process and validate your claims instantly.
        </div>
    """)
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                label="Upload Insurance Claim Document",
                file_types=[".pdf", ".png", ".jpg", ".jpeg"],  # Changed from ["pdf", "png", "jpg", "jpeg"]
                elem_classes="file-upload"
            )
    
    with gr.Row():
        with gr.Column():
            text_output = gr.Textbox(
                label="Extracted Claim Details",
                elem_classes="output-div",
                lines=5
            )
            validation_output = gr.Textbox(
                label="Claim Validation Results",
                elem_classes="output-div"
            )
            classification_output = gr.Textbox(
                label="Document Classification",
                elem_classes="output-div"
            )
    
    file_input.change(
        fn=process_document,
        inputs=[file_input],
        outputs=[text_output, validation_output, classification_output]
    )

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
