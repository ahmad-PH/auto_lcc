import os
import pickle
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing
from iml_group_proj.features.common.data import load
from iml_group_proj.features.common.base import clean_text
from iml_group_proj.features.bert.bert_preprocess import combine_embeddings, get_bert_embeddings
from iml_group_proj.features.common.helper import lower_case, remove_punc, sample_data, tokenize_words, truncate_words, concat_tokens, remove_punc, remove_stop_words, remove_empty_token

OUTPUT_ROOT_PATH = "_output/tfidf_matrix/"
OUTPUT_PATH = "tfidf_{}.pk"

os.makedirs(os.path.join(OUTPUT_ROOT_PATH), exist_ok=True)

title_pipelines = [
        remove_punc,
        lower_case,
        tokenize_words,
        remove_stop_words,
        remove_empty_token,
        truncate_words(50),
        concat_tokens
        ]

sypnosis_pipelines = [
        remove_punc,
        lower_case,
        tokenize_words,
        remove_stop_words,
        remove_empty_token,
        truncate_words(250),
        concat_tokens,
        ] 

def create_and_save_tfidf_features(train, test, bow: bool = True):
    cleaned_train_title = clean_text(train.title, title_pipelines)
    cleaned_train_sypnosis = clean_text(train.sypnosis, sypnosis_pipelines)
    cleaned_test_title = clean_text(test.title, title_pipelines)
    cleaned_test_sypnosis = clean_text(test.sypnosis, sypnosis_pipelines)

    title_vectorizer = CountVectorizer() if bow else TfidfVectorizer()
    sypnosis_vectorizer = CountVectorizer() if bow else TfidfVectorizer()

    train_title_tfidf = title_vectorizer.fit_transform(cleaned_train_title)
    train_sypnosis_tfidf = sypnosis_vectorizer.fit_transform(cleaned_train_sypnosis)
    print(train_title_tfidf.shape)
    print(train_sypnosis_tfidf.shape)
    
    print(f"Saving to {os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format('train'))}")
    with open(os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format("train")), "wb") as f:
        pickle.dump([train_title_tfidf, train_sypnosis_tfidf], f)

    test_title_tfidf = title_vectorizer.transform(cleaned_test_title)
    test_synopsis_tfidf = sypnosis_vectorizer.transform(cleaned_test_sypnosis)

    print(f"Saving to {os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format('test'))}")
    with open(os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format("test")), "wb") as f:
        pickle.dump([test_title_tfidf, test_synopsis_tfidf], f)

classes, train, test = load()
create_and_save_tfidf_features(train, test, bow=False)

print("Done!")
