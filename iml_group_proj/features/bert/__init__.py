import torch
from typing import Any
import pickle
from os import path
from typing import NamedTuple
from sklearn import preprocessing
from iml_group_proj.features.common.data import load
from iml_group_proj.features.common.base import clean_text
from iml_group_proj.features.bert.bert_preprocess import combine_embeddings, get_bert_embeddings
from iml_group_proj.features.common.helper import remove_punc, sample_data, tokenize_words, truncate_words, concat_tokens, remove_punc

title_pipelines = [
        # remove_punc,
        # tokenize_words,
        # truncate_words(50),
        # concat_tokens
        get_bert_embeddings,
        combine_embeddings
        ]

sypnosis_pipelines = [
        # remove_punc,
        tokenize_words,
        truncate_words(250),
        concat_tokens,
        ] + title_pipelines

def get_BERT_features(train, test):
    cleaned_train_title = clean_text(train.title, title_pipelines)
    cleaned_test_title = clean_text(test.title, title_pipelines)

    cleaned_train_sypnosis = clean_text(train.sypnosis, sypnosis_pipelines)
    cleaned_test_sypnosis = clean_text(test.sypnosis, sypnosis_pipelines)

    X_train = torch.concat([cleaned_train_title, cleaned_train_sypnosis], axis=1).detach().numpy()
    X_test = torch.concat([cleaned_test_title, cleaned_test_sypnosis], axis=1).detach().numpy()

    return X_train, X_test

class BertData(NamedTuple):
    train_title_embds: Any = None
    train_sypnosis_embds: Any = None
    test_title_embds: Any = None
    test_sypnosis_embds: Any = None

def load_pickle_bert(file_path: str) -> BertData:
    with open(path.join(file_path, 'bert_train.pk'), 'rb') as f:
        [tr_title_embeddings, tr_synopsis_embeddings] = pickle.load(f)

    with open(path.join(file_path, 'bert_test.pk'), 'rb') as f:
        [te_title_embeddings, te_synopsis_embeddings] = pickle.load(f)

    return BertData(
            tr_title_embeddings,
            tr_synopsis_embeddings,
            te_title_embeddings,
            te_synopsis_embeddings
            )


