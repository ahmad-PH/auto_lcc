
from os import PathLike
import numpy as np
from typing import List
from sklearn.decomposition import TruncatedSVD
from iml_group_proj.data.embeddings import EmbeddingsData, EmbeddingsType


class DataMode:
    title_only = "title"
    synopsis_only = "synopsis"
    both_title_synopsis = "both"

class AverageEmbeddingsPerRecord:
    
    @staticmethod
    def prep(data: EmbeddingsData, embeddings_type: str, data_mode: str=DataMode.both_title_synopsis):

        train_data = (data.train_title, data.train_synopsis)
        test_data = (data.test_title, data.test_synopsis)

        if data_mode == DataMode.title_only:
            train_data = [train_data[0]]
            test_data = [test_data[0]]
        
        if data_mode == DataMode.synopsis_only:
            train_data = [train_data[1]]
            test_data = [test_data[1]]

        ml_data = [train_data, test_data]

        if embeddings_type == EmbeddingsType.TFIDF:
            X_train, X_test = [np.hstack(data) for data in ml_data]
        else:
            X_train, X_test = [np.hstack([_row_avg(d) for d in data]) for data in ml_data]
        y_train, y_test = data.train_labels, data.test_labels

        return X_train, y_train, X_test, y_test

def _row_avg(data: List[np.ndarray]) -> np.ndarray:
    embs_size = data[0].shape[1]
    return np.array([row.mean(axis=0) if row.shape[0] > 0 else np.zeros((embs_size), dtype=np.float32) for row in data])
