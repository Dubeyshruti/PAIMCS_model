Input_DIR = '../../PretrainCorpus'
import os
def pre_process(text: str="") -> str:
    texts = text.split("\n\n")
    temp = []
    for txt in texts:
        if '# Chunk' in txt:
            continue
        txt.replace("[/INST]", "").replace('[INST]', '')
        temp.append(txt)
    return "\t".join(temp)
def main():
    files = (f for f in os.listdir(Input_DIR) if f.endswith(".txt"))
    text = ""
    for fil in files:
        with open(os.path.join(Input_DIR, fil), 'r', encoding='utf-8') as f:
            text += pre_process(f.read())+ "\n"
    with open("../../tokenizer_train_input.txt", 'w', encoding='utf-8') as f:
        f.write(text)
if __name__=="__main__":
    main()
