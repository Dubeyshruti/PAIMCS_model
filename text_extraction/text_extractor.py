from os import listdir, makedirs, path
from PyPDF2 import PdfReader
from json import dump
from tqdm import tqdm
from pandas import Series
PDF_DIR = "/content/ncertBooks/"
PDF_FILES = (f for f in listdir(PDF_DIR) if f.endswith('.pdf'))
NCERT_TEXT = "/content/ncert_text.json"
ERROR_pdfs = list()

def extract_text_from_pdf(pdf_path: str):
    with open(pdf_path, 'rb') as f:
        reader = PdfReader(f)
        text = str()
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
        return text

if __name__ == "__main__":
    pdf_text = dict()
    for pdf_file in tqdm(PDF_FILES, total = len([f for f in listdir(PDF_DIR) if f.endswith('.pdf')]), desc = "Extracting text from pdfs"):
        try:
            pdf_text[pdf_file[:-4]] = extract_text_from_pdf(path.join(PDF_DIR, pdf_file))
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            ERROR_pdfs.append(pdf_file)
    with open(NCERT_TEXT, 'w') as f:
        dump(pdf_text, f, ensure_ascii=False, indent=4)
    if len(ERROR_pdfs) > 0:
        Series(ERROR_pdfs).to_csv('Exceptions_pdfs.csv', index=False)
