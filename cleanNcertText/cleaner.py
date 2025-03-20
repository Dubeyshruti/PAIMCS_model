from asyncio import to_thread, gather, Semaphore, run, create_task, as_completed
from logging import basicConfig, INFO, info, error, warning
from os import path, listdir
from re import search
from typing import List
from prompt import get_prompt
from huggingface_hub import InferenceClient
from tqdm import tqdm

basicConfig(level = INFO, format="%(asctime)s [%(levelname)s] %(message)s") # Configure logging
client = InferenceClient(provider="fireworks-ai", api_key=os.environ.get('HF_TOKEN')) # Initialize client with api token

def text_split(file_path: str) -> List[str]:
    """ Splits the text of the file into three roughly equal parts, trying to break on sentence or whitespace boundaries.
    Args= file_path (str): Path to the text file.
    Returns= List[str]: A list of three text parts. """
    if not path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    length = len(text)
    if length == 0:
        return ["", "", ""]

    part_size = length // 3; parts = []; start = 0
    for _ in range(2): # Find two split points that try to land on punctuation/whitespace boundaries
        end = start + part_size
        if end < length:
            match = search(r'[\.\s।?;,:"]', text[end:])
            if match:
                end += match.start() + 1
        parts.append(text[start:end].strip())
        start = end

    parts.append(text[start:].strip()) # Append the remaining text as the third part
    return parts


def clean_text(dirty_text: str) -> str:
    """ Sends the dirty text to the cleaning API and returns the cleaned text. Args= dirty_text (str): The text to be cleaned.
    Returns= str: The cleaned text. """
    res = client.chat.completions.create(model="perplexity-ai/r1-1776", messages=[{'role': 'user', 'content': get_prompt(dirty_text)}],  temperature=0.6, max_tokens=3072, top_p=0.7)
    return res.choices[0].message.content.split('</think>')[-1].strip()

async def clean_and_save_txt(file_path: str) -> None:
    """ Splits the file text into three parts, cleans each part in parallel using asyncio, joins the cleaned parts, and overwrites the original file with the 
    cleaned text.  Args= file_path (str): Path to the text file. """
    try:
        info(f"Processing file: {file_path}");  parts = text_split(file_path)

        # Use asyncio.to_thread to run blocking I/O (clean_text) in a separate thread.
        tasks = [to_thread(clean_text, part) for part in parts]
        cleaned_parts = await gather(*tasks)
        cleaned_text = "\n".join(cleaned_parts)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        info(f"Successfully cleaned and saved file: {file_path}")
    except Exception as e:
        error(f"Error processing {file_path}: {e}")

async def process_files(file_list: List[str], concurrency: int = 15) -> None:
    """ Processes a list of files with a concurrency limit.
    Args= file_list (List[str]): List of file paths to process. concurrency (int): Maximum number of files to process concurrently."""
    semaphore = Semaphore(concurrency)

    async def sem_task(file: str):
        async with semaphore:
            await clean_and_save_txt(file)

    tasks = [create_task(sem_task(file)) for file in file_list]
    for completed in tqdm(as_completed(tasks), total=len(tasks), desc="Processing files"):
        await completed


def main() -> None:
    base_dir = path.abspath(path.join("..", "..")); files = [path.join(base_dir, f) for f in listdir(base_dir) if f.endswith(".txt")]
    # we assert there are exactly 675 files.
    if len(files) != 675: warning(f"Expected 675 files but found {len(files)} in {base_dir}")
    run(process_files(files))

if __name__ == "__main__":
    main()