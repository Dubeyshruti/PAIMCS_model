BG = 'You are provided with raw text that has been extracted from a PDF. This text includes extra formatting artifacts such as extra newline characters, unwanted Unicode symbols (e.g., "\\n", "\\square"), page numbers, and other noise that disrupts the natural low of the text.'

TASK = '''Your task is to clean and reformat the text according to the following guidelines:
1. Remove all extraneous characters, formatting artifacts, and any noise (e.g., stray Unicode symbols, page numbers, repeated line breaks).
2. Preserve all meaningful content, including section headings, numbered lists, bullet points, formulas, and any domain-specific symbols that are part of the actual text.
3. Ensure that the final output is a coherent, well-organized, and naturally flowing text with clear sentences.
4. Do not remove or alter any information that contributes to the true meaning or context of the text.

Below is the extracted text:
----------------------------'''

TAIL = '''
----------------------------

Please only output the cleaned and well-organized text.'''

def get_prompt(TEXT: str="") -> str:
    return "".join([BG, TASK, TEXT, TAIL])

def main()->None:
    print(get_prompt())

if __name__=='__main__':
    main()
