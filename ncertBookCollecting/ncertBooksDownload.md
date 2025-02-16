### NCERT Book Collection
- link patttern identified manually by visiting [NCERT Website](https://ncert.nic.in/textbook.php?)
- All the pdf urls made by running the script [make_pdf_urls.py]

### Process to Download the pdfs
Create a virtual environment and install libraries
```bash
python -m venv ncert_download_venv
source ncert_download_venv/bin/activate # if running on linux
ncert_download_venv\Scripts\activate # if running on windows
pip install -r ncert_download_requirements.txt
python ncertBooksDownloader.py
```
