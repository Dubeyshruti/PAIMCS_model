from os import listdir, path
from PyPDF2 import PdfReader
from json import dump, load
from tqdm import tqdm
from pandas import Series
from pdf2image import convert_from_path
from pytesseract import image_to_string
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# --- Configuration ---
PDF_DIR = "/content/ncertBooks/"
PDF_FILES = [f for f in listdir(PDF_DIR) if f.endswith('.pdf')]
NCERT_TEXT = "/content/ncert_text.json"
ERROR_pdfs = set()
OPTIMIZED_DPI = 277
OCR_CONFIG = "--oem 1 --psm 1"

# --- Progress Saving Files ---
PROGRESS_FILE = "/content/pdf_extraction_progress.json"  # To save processed PDFs list and current state
EXTRACTED_TEXT_FILE = "/content/partial_ncert_text.json" # To save partially extracted text

def ocr(pdf_file: str) -> dict:
    text = str()
    for image in convert_from_path(pdf_file, dpi = OPTIMIZED_DPI):
        text += image_to_string(image, lang='hin', config=OCR_CONFIG) + '\n'
    return text

def extract_text_from_pdf(pdf_file: str) -> dict:
    global ERROR_pdfs
    pdf_path = path.join(PDF_DIR, pdf_file)
    lang = path.basename(pdf_path).split('_')[2]
    try:
        if lang == 'h':
            return {pdf_file[:-4]: ocr(pdf_path)}
        else:
            try:
                text = str()
                with open(pdf_path, 'rb') as f:
                    reader = PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + '\n'
                return {pdf_file[:-4]: text}
            except:
                return {pdf_file[:-4]: ocr(pdf_path)}
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")
        ERROR_pdfs.add(pdf_file)
        return {pdf_file[:-4]: None}

def save_progress(processed_pdfs, current_pdf_text_dict):
    """Saves the current progress to disk."""
    global ERROR_pdfs
    progress_data = {"processed_pdfs": list(processed_pdfs), "error_pdfs": list(ERROR_pdfs)} # Convert set to list for JSON serializability
    with open(PROGRESS_FILE, 'w') as f_progress:
        dump(progress_data, f_progress, indent=4)
    with open(EXTRACTED_TEXT_FILE, 'w') as f_text:
        dump(current_pdf_text_dict, f_text, ensure_ascii=False, indent=4)

def load_progress():
    """Loads progress from disk if available."""
    processed_pdfs = set()
    current_pdf_text_dict = {}
    try:
        with open(PROGRESS_FILE, 'r') as f_progress:
            progress_data = load(f_progress)
            processed_pdfs = set(progress_data.get("processed_pdfs", [])) # Load as set for efficient checking
            global ERROR_pdfs
            ERROR_pdfs = set(progress_data.get("error_pdfs", []))
        try: # Load partial text, might not exist on first run
            with open(EXTRACTED_TEXT_FILE, 'r') as f_text:
                current_pdf_text_dict = load(f_text)
        except FileNotFoundError:
            pass # No partial text file yet, start with empty dict

    except FileNotFoundError:
        print("No progress file found, starting from scratch.")
    return processed_pdfs, current_pdf_text_dict

if __name__ == "__main__":
    processed_pdfs, pdf_text_dict = load_progress() # Load saved progress

    pdf_files_to_process = [pdf_file for pdf_file in PDF_FILES if pdf_file not in processed_pdfs] # Only process unprocessed PDFs
    print(f"Resuming extraction. {len(processed_pdfs)} PDFs already processed. {len(pdf_files_to_process)} PDFs remaining.")

    if pdf_files_to_process: # Only start pool if there are files to process
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            futures = {executor.submit(extract_text_from_pdf, pdf_file): pdf_file for pdf_file in pdf_files_to_process} # Map future to filename

            for future in tqdm(as_completed(futures), total=len(pdf_files_to_process), desc="Extracting text from PDFs"):
                pdf_file = futures[future] # Get original filename from future
                partial_result = future.result()
                pdf_text_dict.update(partial_result)
                processed_pdfs.add(pdf_file) # Mark as processed
                save_progress(processed_pdfs, pdf_text_dict) # SAVE PROGRESS AFTER EACH PDF

    # Final output saving and error reporting
    with open(NCERT_TEXT, 'w') as f:
        dump(pdf_text_dict, f, ensure_ascii=False, indent=4)

    if ERROR_pdfs:
        Series(list(ERROR_pdfs)).to_csv('Exceptions_pdfs.csv', index=False, header=['pdfName'])
    print("\nText extraction completed.")