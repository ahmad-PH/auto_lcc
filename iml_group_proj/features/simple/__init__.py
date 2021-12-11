import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing
from iml_group_proj.features.common.data import load
from iml_group_proj.features.common.base import clean_text
from iml_group_proj.features.bert.bert_preprocess import combine_embeddings, get_bert_embeddings
from iml_group_proj.features.common.helper import remove_punc, sample_data, tokenize_words, truncate_words, concat_tokens, remove_punc, remove_stop_words, remove_empty_token

title_pipelines = [
        remove_punc,
        tokenize_words,
        remove_stop_words,
        remove_empty_token,
        truncate_words(50),
        concat_tokens
        ]

sypnosis_pipelines = [
        remove_punc,
        tokenize_words,
        remove_stop_words,
        remove_empty_token,
        truncate_words(250),
        concat_tokens,
        ] 

def get_simple_features(train, test, bow: bool = True, only_title:bool = True):
    # TODO: Add functionality to get only title/sypnosis optimally
    cleaned_train_title = clean_text(train.title, title_pipelines)
    cleaned_test_title = clean_text(test.title, title_pipelines)

    vectorizer = CountVectorizer() if bow else TfidfVectorizer()

    X_train = vectorizer.fit_transform(cleaned_train_title)
    X_test = vectorizer.transform(cleaned_test_title)

    if only_title:
        return X_train, X_test

    cleaned_train_sypnosis = clean_text(train.sypnosis, sypnosis_pipelines)
    cleaned_test_sypnosis = clean_text(test.sypnosis, sypnosis_pipelines)

    vectorizer = CountVectorizer() if bow else TfidfVectorizer()
    X_train_2 = vectorizer.fit_transform(cleaned_train_sypnosis)
    X_test_2 = vectorizer.transform(cleaned_test_sypnosis)

    X_train = hstack((X_train, X_train_2))
    X_test = hstack((X_test, X_test_2))

    return X_train, X_test
