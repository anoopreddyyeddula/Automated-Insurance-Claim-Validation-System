---
title: Automated Insurance Claim Validation System
emoji: 📄
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.27.0
app_file: app.py
pinned: false
license: mit
---

# Automated Insurance Claim Validation System

This project automates the validation process for insurance claims using image processing and NLP techniques.

## Features:
- Support for both PDF and image files
- Multiple page processing
- EasyOCR for reliable text extraction
- BERT-based text validation
- Document classification using pre-trained models
- Export results to Excel
- User-friendly Gradio interface

## System Requirements:
- Python 3.9+
- Required Python packages (see requirements.txt)
- System dependencies: tesseract-ocr, poppler-utils

## Installation:
```bash
pip install -r requirements.txt
```

## Usage:
1. Upload an insurance claim document (PDF or image)
2. The system will:
   - Process all pages in the document
   - Extract text using EasyOCR
   - Validate the extracted text
   - Classify each page
   - Generate a downloadable report
3. Download the Excel report for detailed analysis

## Models Used:
- OCR: EasyOCR
- Text Classification: DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
- Document Classification: Donut (naver-clova-ix/donut-base-finetuned-rvlcdip)

## Live Demo:
Access the live demo at: https://huggingface.co/spaces/anoopreddyyeddula/Automated-Insurance-Claim-Validation-System