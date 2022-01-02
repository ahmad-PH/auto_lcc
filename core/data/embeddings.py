import os
import os
import pickle
import scipy
import numpy as np
from abc import ABC, abstractmethod
from typing import NamedTuple, List, Union, Tuple

from iml_group_proj.data.utils import load_libofc_df


EMBEDDINGS_TYPE = List[np.ndarray]
# EMBEDDINGS_TYPE = Union[List[np.ndarray], scipy.sparse]
class EmbeddingsData(NamedTuple):
    train_title: EMBEDDINGS_TYPE
    train_synopsis: EMBEDDINGS_TYPE
    train_labels: np.ndarray
    test_title: EMBEDDINGS_TYPE
    test_synopsis: EMBEDDINGS_TYPE
    test_labels: np.ndarray



class EmbeddingsType: 
    W2V = "w2v"
    BERT_XS = "bert_tiny"
    BERT_S = "bert_small"
    BERT_L = "bert_large"
    TFIDF = "tfidf"

class EmbeddingsDataLoader:
    @staticmethod
    def load(file_path: os.PathLike, e_type: str):
        _, train_df, test_df = load_libofc_df(os.path.join(file_path, "trainTest.pk"))

        if "bert" in e_type:
            data = _load_bert_embeddings(file_path, (train_df.shape[0], test_df.shape[0]), e_type)
        else:
            data = _load_other_embeddings(file_path, e_type)


        return EmbeddingsData(
                data[0],
                data[1],
                train_df["class"].values,
                data[2],
                data[3],
                test_df["class"].values
                )

def _load_other_embeddings(file_path, e_type):
    with open(os.path.join(file_path, f"{e_type}_train.pk"), 'rb') as f:
        [tr_title_embeddings, tr_synopsis_embeddings] = pickle.load(f)

    with open(os.path.join(file_path, f"{e_type}_test.pk"), 'rb') as f:
        [te_title_embeddings, te_synopsis_embeddings] = pickle.load(f)


    return (
            tr_title_embeddings,
            tr_synopsis_embeddings,
            te_title_embeddings,
            te_synopsis_embeddings,
            )


def _load_bert_embeddings(file_path: os.PathLike, expected_length: Tuple[int,int], e_type):
    old_file_name_format = os.path.join(file_path, f"{e_type}_train.pk")
    if os.path.isfile(old_file_name_format):
        return _load_other_embeddings(file_path, e_type)

    folder_path = os.path.join(file_path, e_type)

    tr_title_embeddings = _load_each_bert_embs(folder_path, "title_train", expected_length[0])
    tr_synop_embeddings = _load_each_bert_embs(folder_path, "synopsis_train", expected_length[0])
    te_title_embeddings = _load_each_bert_embs(folder_path, "title_test", expected_length[1])
    te_synop_embeddings = _load_each_bert_embs(folder_path, "synopsis_test", expected_length[1])

    return (tr_title_embeddings, tr_synop_embeddings, te_title_embeddings, te_synop_embeddings)


def _load_each_bert_embs(file_path, bert_e_type, expected_length):
    file_path = os.path.join(file_path, bert_e_type)
    all_embeddings = [f_name for f_name in os.listdir(file_path) if "pk" in f_name]
    all_embeddings.sort()
    assert len(all_embeddings) == expected_length, \
    f"{bert_e_type} not fully downloaded yet ({len(all_embeddings)}/{expected_length}). \n Please use runner/build_bert_embeddings.py to download the data first."

    embs = []
    # Do the loading 
    for embeddings_file in all_embeddings[:5]:
        emb_file_name = os.path.join(file_path, embeddings_file)
        print(emb_file_name)
        with open(emb_file_name, "rb") as f:
            embs.append(pickle.load(f))
    return embs
    



