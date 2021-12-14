import os
import os
import pickle
import scipy
import numpy as np
from abc import ABC, abstractmethod
from typing import NamedTuple, List, Union

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
    BERT = "bert"
    TFIDF = "tfidf"

class EmbeddingsDataLoader:

    @staticmethod
    def load(file_path: os.PathLike, e_type: str):
        with open(os.path.join(file_path, f"{e_type}_train.pk"), 'rb') as f:
            [tr_title_embeddings, tr_synopsis_embeddings] = pickle.load(f)

        with open(os.path.join(file_path, f"{e_type}_test.pk"), 'rb') as f:
            [te_title_embeddings, te_synopsis_embeddings] = pickle.load(f)

        _, train_df, test_df = load_libofc_df(os.path.join(file_path, "trainTest.pk"))

        return EmbeddingsData(
                tr_title_embeddings,
                tr_synopsis_embeddings,
                train_df["class"].values,
                te_title_embeddings,
                te_synopsis_embeddings,
                test_df["class"].values
                )
