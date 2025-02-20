Used PyPDF2 and pytesseract for extracting texts from ncertBooks
Set up of tesseract
```bash
sudo apt update
sudo apt install tesseract-ocr poppler-utils tesseract-ocr-hin
tesseract --list-langs
pip install -r text_extraction_requirements.txt
python text_extractor.py
```
