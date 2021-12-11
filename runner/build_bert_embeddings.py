import os
from tqdm import tqdm
import time
import pickle
from typing import List

from transformers import BertModel, BertTokenizer
from iml_group_proj.features.common.data import load

OUTPUT_ROOT_PATH = "_output/bert_embeddings_2"
OUTPUT_PATH = "bert_{}.pk"
BERT_MODEL_NAME = "prajjwal1/bert-tiny"
MAX_TOKENS_LENGTH = 512 # Can effect the sypnosis, check.
VERBOSE = True

classes, train, test = load()
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
model = BertModel.from_pretrained(BERT_MODEL_NAME)

def vprint(s):
    if VERBOSE:
        print(s)

def from_sentences_to_bert_embeddings(sentences: List[str]):
    outputs = []
    for sentence in tqdm(sentences):
        inputs = bert_tokenizer(sentence, truncation=True, max_length=MAX_TOKENS_LENGTH, return_tensors="pt")
        output = model(**inputs)
        outputs.append(output.last_hidden_state.squeeze(0))

    return outputs

vprint("Getting BERT embeddings for training set.")
os.makedirs(os.path.join(OUTPUT_ROOT_PATH), exist_ok=True)

sypnosis_tasks = [(train.sypnosis[:10000], "train_sypnosis_10k"), (train.sypnosis[10000:20000], "train_sypnosis_20k"), (train.sypnosis[20000:], "train_sypnosis_rest")]
tasks = [(train.title, "train_title"), (test.title, "test_title"), (test.sypnosis, "test_sypnosis")] + sypnosis_tasks
for (saving_data, file_name) in tasks:
    file_path = os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format(file_name))
    print(f"Doing {file_path}")
    if os.path.isfile(file_path):
        print("File found, skipping...")
        continue # Skip to the next data

    data = from_sentences_to_bert_embeddings(saving_data)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

# Combining the splitted sypnosis because too big for my machine :|
sypnosis_list = []
for (_, sypnosis_file) in sypnosis_tasks:
    with open(os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format(sypnosis_file)), "rb") as f:
        sypnosis_list.extend(pickle.load(f))

    print(len(sypnosis_list))
    with open(os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format("train_sypnosis")), "wb") as f:
        pickle.dump(sypnosis_list, f)

for title, sypnosis in [("train_title", "train_sypnosis"), ("test_title", "test_sypnosis")]:
    with open(os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format(title)), "rb") as f:
        title_data = pickle.load(f)
    with open(os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format(sypnosis)), "rb") as f:
        sypnosis_data = pickle.load(f)

    assert(len(title_data) == len(sypnosis_data)), f"Title lengths({len(title_data)}) is not equal to sypnosis's({len(sypnosis_data)})"

    with open(os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format(title.split("_")[0])), "wb") as f:
        pickle.dump([title_data, sypnosis_data], f)
    print(f'DONE!, saved at {os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format(title.split("_")[0]))}')

    # os.remove(os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format(title))
    # os.remove(os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format(sypnosis))
