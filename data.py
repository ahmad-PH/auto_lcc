from typing import *
import pickle
import os.path as path
import numpy as np

class Record:
    def __init__(self, cls, title, synopsis, id):
        self.cls: str = cls
        self.cls_idx: int = RecordStore.cls_to_idx_map[cls]
        self.title: str = title
        self.synopsis: str = synopsis
        self.id: int = id
            
        self.title_embeddings_bert: List[Any] = None
        self.synopsis_embeddings_bert: List[Any] = None

        self.title_embeddings_w2v: List[Any] = None
        self.synopsis_embeddings_w2v: List[Any] = None

        self.title_embeddings_tfidf: List[Any] = None
        self.synopsis_embeddings_tfidf: List[Any] = None

    def __str__(self) -> str:
        return "class: {}, title: {}, id:{}, synopsis:{}".format(self.cls, self.title, self.id, self.synopsis)

class RecordStore:
    train_records: List[Record] = []
    test_records: List[Record] = []
    DATA_ROOT = path.join(path.dirname(__file__), "github_data")
    classes: List[str] = []
    cls_to_idx_map: Mapping[str, int] = {}

    """
    Loads the records, without their embeddings. Should always be used before accessing any of the other functions 
    or properties of this class.
    """
    @staticmethod
    def load():
        with open(path.join(RecordStore.DATA_ROOT, 'trainTest.pk'), 'rb') as f:
            classes = pickle.load(f)
            train = pickle.load(f)
            test = pickle.load(f)

        RecordStore.cls_to_idx_map = {v:i for i, v in enumerate(classes)} # Must create the mapping before creating any Records
        RecordStore.classes = classes
        RecordStore.train_records = [Record(cls, title, synopsis, id) for (cls, title, synopsis, id) in train]
        RecordStore.test_records = [Record(cls, title, synopsis, id) for (cls, title, synopsis, id) in test]

    
    """
    Loads the bert embededdings into memory. Only use if you need these embeddings.
    """
    @staticmethod
    def load_bert():
        with open(path.join(RecordStore.DATA_ROOT, 'bert_train.pk'), 'rb') as f:
            [title_embeddings, synopsis_embeddings] = pickle.load(f)

        for i, record in enumerate(RecordStore.train_records):
            record.title_embeddings_bert = title_embeddings[i]
            record.synopsis_embeddings_bert = synopsis_embeddings[i]

        with open(path.join(RecordStore.DATA_ROOT, 'bert_test.pk'), 'rb') as f:
            [title_embeddings, synopsis_embeddings] = pickle.load(f)

        for i, record in enumerate(RecordStore.test_records):
            record.title_embeddings_bert = title_embeddings[i]
            record.synopsis_embeddings_bert = synopsis_embeddings[i]

    @staticmethod
    def _ensure_non_empty(embedding):
        if embedding.shape == (0,):
            return np.zeros((1,300), dtype=np.float32)
        else:
            return embedding

    """
    Loads the word2vec embededdings into memory. Only use if you need these embeddings.
    """
    @staticmethod
    def load_w2v():
        with open(path.join(RecordStore.DATA_ROOT, 'w2v_train.pk'), 'rb') as f:
            [title_embeddings, synopsis_embeddings] = pickle.load(f)

        for i, record in enumerate(RecordStore.train_records):
            record.title_embeddings_w2v = RecordStore._ensure_non_empty(title_embeddings[i])
            record.synopsis_embeddings_w2v = RecordStore._ensure_non_empty(synopsis_embeddings[i])

        with open(path.join(RecordStore.DATA_ROOT, 'w2v_test.pk'), 'rb') as f:
            [title_embeddings, synopsis_embeddings] = pickle.load(f)

        for i, record in enumerate(RecordStore.test_records):
            record.title_embeddings_w2v = RecordStore._ensure_non_empty(title_embeddings[i])
            record.synopsis_embeddings_w2v = RecordStore._ensure_non_empty(synopsis_embeddings[i])

    """
    Loads the word2vec embededdings into memory. Only use if you need these embeddings.
    """
    @staticmethod
    def load_tfidf():
        with open(path.join(RecordStore.DATA_ROOT, 'tfidf_train.pk'), 'rb') as f:
            [title_embeddings, synopsis_embeddings] = pickle.load(f)

        for i, record in enumerate(RecordStore.train_records):
            record.title_embeddings_tfidf = title_embeddings[i]
            record.synopsis_embeddings_tfidf = synopsis_embeddings[i]

        with open(path.join(RecordStore.DATA_ROOT, 'tfidf_test.pk'), 'rb') as f:
            [title_embeddings, synopsis_embeddings] = pickle.load(f)

        for i, record in enumerate(RecordStore.test_records):
            record.title_embeddings_tfidf = title_embeddings[i]
            record.synopsis_embeddings_tfidf = synopsis_embeddings[i]


