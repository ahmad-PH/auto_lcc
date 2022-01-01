import os
import pickle
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing
from iml_group_proj.config import RANDOM_STATE
from iml_group_proj.data.utils import load_libofc_df
from iml_group_proj.data.text_clean import clean_text
from iml_group_proj.data.text_clean.function import lower_case, remove_punc, sample_data, tokenize_words, truncate_words, concat_tokens, remove_punc, remove_stop_words, remove_empty_token
from sklearn.decomposition import TruncatedSVD

OUTPUT_ROOT_PATH = "_output/tfidf_matrix_2/"
OUTPUT_PATH = "tfidf_{}.pk"

os.makedirs(os.path.join(OUTPUT_ROOT_PATH), exist_ok=True)

title_pipelines = [
        remove_punc,
        lower_case,
        tokenize_words,
        remove_stop_words,
        remove_empty_token,
        # truncate_words(50),
        concat_tokens
        ]

synopsis_pipelines = [
        remove_punc,
        lower_case,
        tokenize_words,
        remove_stop_words,
        remove_empty_token,
        # truncate_words(250),
        concat_tokens,
        ] 

def create_and_save_tfidf_features(train, test, bow: bool = True):
    print("Applying text cleaning pipelines...")
    cleaned_train_title = clean_text(train.title, title_pipelines)
    cleaned_train_synopsis = clean_text(train.synopsis, synopsis_pipelines)
    cleaned_test_title = clean_text(test.title, title_pipelines)
    cleaned_test_synopsis = clean_text(test.synopsis, synopsis_pipelines)
    
    print("Applying TFIDF Vectorizer to training set")
    title_vectorizer = CountVectorizer() if bow else TfidfVectorizer()
    synopsis_vectorizer = CountVectorizer() if bow else TfidfVectorizer()

    train_title_tfidf = title_vectorizer.fit_transform(cleaned_train_title)
    train_synopsis_tfidf = synopsis_vectorizer.fit_transform(cleaned_train_synopsis)

    print("Applying dimension reduction (SVD) to training set")
    title_svd = TruncatedSVD(n_components=500, random_state=RANDOM_STATE)
    synopsis_svd = TruncatedSVD(n_components=500, random_state=RANDOM_STATE)

    train_title_tfidf = title_svd.fit_transform(train_title_tfidf)
    train_synopsis_tfidf = synopsis_svd.fit_transform(train_synopsis_tfidf)
    
    print(f"Saving to {os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format('train'))}")
    with open(os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format("train")), "wb") as f:
        pickle.dump([train_title_tfidf, train_synopsis_tfidf], f)

    print("Applying TFIDF Vectorizer to test set")
    test_title_tfidf = title_vectorizer.transform(cleaned_test_title)
    test_synopsis_tfidf = synopsis_vectorizer.transform(cleaned_test_synopsis)

    print("Applying dimension reduction (SVD) to test set")
    test_title_tfidf = title_svd.transform(test_title_tfidf)
    test_synopsis_tfidf = synopsis_svd.transform(test_synopsis_tfidf)

    print(f"Saving to {os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format('test'))}")
    with open(os.path.join(OUTPUT_ROOT_PATH, OUTPUT_PATH.format("test")), "wb") as f:
        pickle.dump([test_title_tfidf, test_synopsis_tfidf], f)

if __name__  == "__main__":
    classes, train, test = load_libofc_df("github_data/trainTest.pk")
    create_and_save_tfidf_features(train, test, bow=False)
    print("Done!")
