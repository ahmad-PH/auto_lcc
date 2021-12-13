import os
import time
import pickle
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
from transformers import BertModel, BertTokenizer

DATA_PATH = "github_data"
BERT_MODEL_NAME = "prajjwal1/bert-small"
OUTPUT_ROOT_PATH = f"_output/{BERT_MODEL_NAME.split('/')[1].replace('-', '_')}_output/"

# BERT_MODEL_NAME = "prajjwal1/bert-tiny"
# BERT_MODEL_NAME = "prajjwal1/bert-medium"

OUTPUT_PATH = "bert_{}.pk"
MAX_TOKENS_LENGTH = 512 # Can effect the synopsis, check.
VERBOSE = True

def tuple_to_df(data: List[Tuple]) -> pd.DataFrame:
    return pd.DataFrame(data, columns=["class", "title", "synopsis", "id"])

with open(os.path.join(DATA_PATH, "trainTest.pk"), 'rb') as f:
    classes = pickle.load(f)
    train = tuple_to_df(pickle.load(f))
    test = tuple_to_df(pickle.load(f))

bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
model = BertModel.from_pretrained(BERT_MODEL_NAME)

def vprint(s):
    if VERBOSE:
        print(s)

def get_path(file_name):
    return os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format(file_name))

def from_sentences_to_bert_embeddings(sentences: List[str]):
    outputs = []
    for sentence in tqdm(sentences):
        inputs = bert_tokenizer(sentence, truncation=True, max_length=MAX_TOKENS_LENGTH, return_tensors="pt")
        output = model(**inputs)
        outputs.append(output.last_hidden_state.squeeze(0).detach().numpy())

    return outputs

vprint("Getting BERT embeddings...")
os.makedirs(OUTPUT_ROOT_PATH, exist_ok=True)

# for (titles, synopsis, name) in [(test.title, test.synopsis, "test"),(train.title, train.synopsis, "train")]:
for (titles, synopsis, name) in [(train.title, train.synopsis, "train")]:
    title_embeddings = from_sentences_to_bert_embeddings(titles)
    synopsis_embeddings = from_sentences_to_bert_embeddings(synopsis)
    assert len(title_embeddings) == len(synopsis_embeddings), f"Title lengths({len(title_embeddings)}) is not equal to synopsis's({len(synopsis_embeddings)})"

    with open(os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format(name)), "wb") as f:
        pickle.dump([title_embeddings, synopsis_embeddings], f)

    vprint(f'DONE!, saved at {os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format(name))}')
