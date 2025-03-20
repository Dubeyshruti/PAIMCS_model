"For saving Extracted text content into txt files."
from datasets import load_dataset
def main()->None:
    ds = load_dataset("ritik22912/ncert_upsc_text_6to12")["train"].to_dict()
    for name, text in zip(ds['pdfName'], ds['text']):
        with open(f"../../{name}.txt", 'w') as f:
            f.write(text)
if __name__=="__main__":
    main()