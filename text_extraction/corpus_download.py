import argparse
import os
import re
from typing import List
import shutil
import zipfile
from datasets import load_dataset
from tqdm import tqdm

# === CONFIGURATION ===
max_file_size_bytes = 2048 * 1024 * 1024  # 2GB per .txt
max_files_per_batch = 10                 # Zip every 10 .txt files
max_batch_unzipped_bytes = 20 * 1024 * 1024 * 1024  # Zip if uncompressed batch hits 20GB

# === Sentence Splitter ===
def split_sentences(texts: List[str]) -> List[str]:
    pat = re.compile(r'(?<=[\.!?।])\s+')
    sentences = []
    for doc in texts:
        for sent in pat.split(doc):
            sent = sent.strip()
            if sent:
                sentences.append(sent)
    return sentences

# === Initialize File State ===
def open_new_file(output_dir, file_index):
    file_path = os.path.join(output_dir, f"part_{file_index:03}.txt")
    f = open(file_path, "w", encoding="utf-8")
    return f, file_path

# === Zip and Cleanup ===
def zip_and_cleanup_batch(batch_files, batch_index, output_dir):
    if not batch_files:
        return 0  # nothing zipped

    zip_name = os.path.join(output_dir, f"batch_{batch_index:03}.zip")
    print(f"\n📦 Zipping batch {batch_index} with {len(batch_files)} files...")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for path in tqdm(batch_files, desc="Zipping", unit="file"):
            arcname = os.path.basename(path)
            zipf.write(path, arcname)
            os.remove(path)
    print(f"✅ Batch {batch_index} zipped and cleaned.\n")
    return 1  # one batch zipped

# === Write Dataset ===
def write_dataset_sentences(dataset, text_column, output_dir, label=""):
    os.makedirs(output_dir, exist_ok=True)

    file_index = 0
    batch_index = 0
    current_file_size = 0
    current_batch_size = 0
    batch_files = []

    current_file, current_file_path = open_new_file(output_dir, file_index)

    for example in tqdm(dataset, desc=f"Processing {label}", unit="doc"):
        raw_text = str(example[text_column])
        sentence_list = split_sentences([raw_text])

        for sentence in sentence_list:
            sentence_line = sentence.strip() + "\n"
            sentence_size = len(sentence_line.encode("utf-8"))

            if current_file_size + sentence_size > max_file_size_bytes:
                current_file.close()
                batch_files.append(current_file_path)
                current_batch_size += current_file_size

                file_index += 1
                current_file, current_file_path = open_new_file(output_dir, file_index)
                current_file_size = 0

                # Check if it's time to zip
                if len(batch_files) >= max_files_per_batch or current_batch_size >= max_batch_unzipped_bytes:
                    if zip_and_cleanup_batch(batch_files, batch_index, output_dir):
                        batch_index += 1
                        batch_files = []
                        current_batch_size = 0

            current_file.write(sentence_line)
            current_file_size += sentence_size

    # Final cleanup
    if not current_file.closed:
        current_file.close()
        batch_files.append(current_file_path)
        current_batch_size += current_file_size

    if batch_files:
        zip_and_cleanup_batch(batch_files, batch_index, output_dir)

    print("\n🎉 All done for:", label)

# === Main ===
def main():
    parser = argparse.ArgumentParser(description="Process dataset by language")
    parser.add_argument("--lang", required=True, choices=["hi", "en"], help="Language: 'hi' or 'en'")
    args = parser.parse_args()

    if args.lang == "hi":
        dataset = load_dataset("Hindi-data-hub/odaigen_hindi_pre_trained_sp", split="train", streaming=True)
        text_column = "content"
        label = "Hindi Dataset"
        output_dir = "../../corpus_batches_hi"
    else:
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        text_column = "text"
        label = "English Dataset"
        output_dir = "../../corpus_batches_en"

    write_dataset_sentences(dataset, text_column, output_dir, label=label)

if __name__ == "__main__":
    main()
