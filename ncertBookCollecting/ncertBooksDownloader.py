from requests import get
from pandas import read_csv, DataFrame
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import makedirs
from random import choice
import time

save_dir = "ncertBooks"
makedirs(save_dir, exist_ok=True)
pdf_urls = read_csv("pdf_urls.csv")
referrers = ['https://www.google.com', 'https://www.yahoo.com', 'https://www.duckduckgo.com', 'https://www.perplexity.ai']
not_downloaded_pdfs = {'pdfName': [], 'downloadLink': []}  # Initialize as empty lists

def downloadPdf(pdfName, downloadLink):
    try:
        response = get(downloadLink, headers={'Referrer': choice(referrers)}, timeout=180)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        with open(f"{save_dir}/{pdfName}", 'wb') as f:
            f.write(response.content)
        return None  # Indicate success
    except Exception as e:
        print(f"\nError downloading {pdfName} from {downloadLink}: {e}") # More informative error message
        return pdfName, downloadLink  # Return the failed PDF info

def main():
    with ThreadPoolExecutor(max_workers=6) as executor:  # Create the executor *once*
        futures = {executor.submit(downloadPdf, row.pdfName, row.downloadLink): (row.pdfName, row.downloadLink) for _, row in pdf_urls.iterrows()}
        for future in tqdm(as_completed(futures), total=len(pdf_urls), desc="Downloading PDFs"):
            result = future.result()
            if result:  # Check if download failed
                pdfName, downloadLink = result
                not_downloaded_pdfs['pdfName'].append(pdfName)
                not_downloaded_pdfs['downloadLink'].append(downloadLink)


    if len(not_downloaded_pdfs) > 0:
        DataFrame(not_downloaded_pdfs).to_csv("ncertBooks_download_exceptions.csv", index=False) #Added index=False to avoid index column in csv

if __name__ == "__main__":
    main()
