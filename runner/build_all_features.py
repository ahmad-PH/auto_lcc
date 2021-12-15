import os
from runner import build_tfidf_features
from runner.build_bert_embeddings import build_bert_embeddings
    

if __name__ == "__main__":

    if not os.path.isfile("github_data/tfidf_train.pk"):
        print("Building tfidf features...")
        build_tfidf_features()
    else:
        print("tfidf features found, skipping")

    print("Building BERT embeddings (tiny) features...")
    build_bert_embeddings("tiny")

    print("Building BERT embeddings (small) features...")
    build_bert_embeddings("small")

    print("Building BERT embeddings (large) features...")
    build_bert_embeddings("large")
