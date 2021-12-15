import os
import time
import click
import shutil
import pickle
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
from transformers import BertModel, BertTokenizer

DATA_PATH = "github_data"

BERT_NAME_MAPPER = {
        "tiny":  "prajjwal1/bert-tiny",
        "small":  "prajjwal1/bert-small",
        "large":  "bert-base-uncased",
        }

BERT_SIZENUM_MAPPER = {
        "tiny": 1.0,
        "small":4.1,
        "large":10,
        }

BERT_SIZE_MAPPER = {
        "tiny": "1.00GB",
        "small":  "3.99GiB",
        "large":  "5.99GB",
        }

OUTPUT_PATH = "bert_{}.pk"
MAX_TOKENS_LENGTH = 512 # Can effect the synopsis, check.
VERBOSE = True

def tuple_to_df(data: List[Tuple]) -> pd.DataFrame:
    return pd.DataFrame(data, columns=["class", "title", "synopsis", "id"])

with open(os.path.join(DATA_PATH, "trainTest.pk"), 'rb') as f:
    classes = pickle.load(f)
    train = tuple_to_df(pickle.load(f))
    test = tuple_to_df(pickle.load(f))

# Code from: https://www.codegrepper.com/code-examples/python/convert+bytes+to+human+readable+python
def human_readable_size(size, decimal_places=3):
    for unit in ['B','KiB','MiB','GiB','TiB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f}{unit}"

@click.command()
@click.option("--model_size", default="small", help="size of BERT model. expected [tiny, small, large]")
def build_bert_embeddings(model_size: str):

    assert model_size in ["tiny", "small", "large"], f"{model_size} not expected, expected [tiny, small, large]"

    print("Getting BERT embeddings...")
    OUTPUT_ROOT_PATH = os.path.join("github_data", f"bert_{model_size}")
    os.makedirs(OUTPUT_ROOT_PATH, exist_ok=True)

    disk_stats = shutil.disk_usage(OUTPUT_ROOT_PATH)
    downloaded_size = sum(os.path.getsize(f) for f in os.listdir(OUTPUT_ROOT_PATH) if os.path.isfile(f))
    print(f"{BERT_NAME_MAPPER[model_size]} requires a total of {BERT_SIZE_MAPPER[model_size]}: ")
    if downloaded_size > 0:
        print(f"Downloaded {human_readable_size(downloaded_size)}")

    if disk_stats.free / (1024 * 1024 * 1024) < BERT_SIZENUM_MAPPER[model_size]:
        print(f"You have only {human_readable_size(disk_stats.free)} disk space left.")
        ans = input("Space won't be enough. Are you sure you want to continue loading the embeddings? [y/N]")
        if ans.lower() != "y":
            print("Exiting the program")
            return
    
    print(f"Downloading ({BERT_NAME_MAPPER[model_size]}) to {OUTPUT_ROOT_PATH}")
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_NAME_MAPPER[model_size])
    model = BertModel.from_pretrained(BERT_NAME_MAPPER[model_size])

    def get_file_path(file_name, sentence_id):
        return os.path.join(file_name, f"{sentence_id:06d}.pk")

    def from_sentences_to_bert_embeddings(sentences: List[str], root_output_path: str):
        for idx, sentence in tqdm(enumerate(sentences), total=len(sentences)):
            outputting_file_path = get_file_path(root_output_path, idx)
            if os.path.isfile(outputting_file_path):
                continue

            inputs = bert_tokenizer(sentence, truncation=True, max_length=MAX_TOKENS_LENGTH, return_tensors="pt")
            output = model(**inputs).last_hidden_state.squeeze(0).detach().numpy()

            with open(outputting_file_path, "wb") as f:
                pickle.dump(output, f)

        return True
    for (titles, synopsis, name) in [(train.title, train.synopsis, "train"), (test.title, test.synopsis, "test")]:
        title_folder_name = os.path.join(OUTPUT_ROOT_PATH, f"title_{name}")
        synopsis_folder_name = os.path.join(OUTPUT_ROOT_PATH, f"synopsis_{name}")
        os.makedirs(title_folder_name, exist_ok=True)
        os.makedirs(synopsis_folder_name, exist_ok=True)

        from_sentences_to_bert_embeddings(titles, title_folder_name)
        print(f'DONE!, embeddings saved at {title_folder_name}')
        from_sentences_to_bert_embeddings(synopsis, synopsis_folder_name)
        print(f'DONE!, embeddings saved at {synopsis_folder_name}')

if __name__ == "__main__":
    build_bert_embeddings()
