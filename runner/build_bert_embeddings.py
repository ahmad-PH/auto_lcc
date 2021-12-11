import os
import time
import pickle
from typing import List

from transformers import BertModel, BertTokenizer
from iml_group_proj.features.common.data import load

OUTPUT_ROOT_PATH = "_output/bert_embeddings"
OUTPUT_PATH = "bert_{}.pk"
BERT_MODEL_NAME = "prajjwal1/bert-tiny"
MAX_TOKENS_LENGTH = 512 # Can effect the sypnosis, check.
VERBOSE = True

classes, train, test = load()
# train = train[:5]
# test = test[:5]
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
model = BertModel.from_pretrained(BERT_MODEL_NAME)

def vprint(s):
    if VERBOSE:
        print(s)

def from_sentences_to_bert_embeddings(sentences: List[str]):
    outputs = []
    for sentence in sentences:
        inputs = bert_tokenizer(sentence, truncation=True, max_length=MAX_TOKENS_LENGTH, return_tensors="pt")
        output = model(**inputs)
        outputs.append(output.last_hidden_state.squeeze(0))

    return outputs

vprint("Getting BERT embeddings for training set.")
start = time.time()
train_data = [from_sentences_to_bert_embeddings(sentences) for sentences in [train.title, train.sypnosis]]
vprint(f"Done! [{time.time() - start:.3f}s], getting BERT embeddings for training set.")
start = time.time()
test_data = [from_sentences_to_bert_embeddings(sentences) for sentences in [test.title, test.sypnosis]]
vprint(f"Done! [{time.time() - start:.3f}s], saving as .pickle files")

os.makedirs(os.path.join(OUTPUT_ROOT_PATH), exist_ok=True)

with open(os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format("train")), "wb") as f:
    pickle.dump(train_data, f)

with open(os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format("test")), "wb") as f:
    pickle.dump(test_data, f)

vprint(f"Embeddings stored successfully in {OUTPUT_ROOT_PATH}")
