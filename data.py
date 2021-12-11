from typing import *
import pickle
import os.path as path

class Record:
    def __init__(self, cls, title, synopsis, id):
        self.cls: str = cls
        self.title: str = title
        self.synopsis: str = synopsis
        self.id: int = id
            
        self.title_embeddings_bert: List[Any] = None
        self.synopsis_embeddings_bert: List[Any] = None

        self.title_embeddings_w2v: List[Any] = None
        self.synopsis_embeddings_w2v: List[Any] = None

    def __str__(self) -> str:
        return "class: {}, title: {}, id:{}, synopsis:{}".format(self.cls, self.title, self.id, self.synopsis)
        
class RecordStore:
    train_records: List[Record] = []
    test_records: List[Record] = []
    DATA_ROOT = path.join(".", "github_data")

    @staticmethod
    def load():
        with open(path.join(RecordStore.DATA_ROOT, 'trainTest.pk'), 'rb') as f:
            classes = pickle.load(f)
            train = pickle.load(f)
            test = pickle.load(f)

        RecordStore.train_records = [Record(cls, title, synopsis, id) for (cls, title, synopsis, id) in train]
        RecordStore.test_records = [Record(cls, title, synopsis, id) for (cls, title, synopsis, id) in test]
    
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
    def load_w2v():
        with open(path.join(RecordStore.DATA_ROOT, 'w2v_train.pk'), 'rb') as f:
            [title_embeddings, synopsis_embeddings] = pickle.load(f)

        for i, record in enumerate(RecordStore.train_records):
            record.title_embeddings_w2v = title_embeddings[i]
            record.synopsis_embeddings_w2v = synopsis_embeddings[i]

        with open(path.join(RecordStore.DATA_ROOT, 'w2v_test.pk'), 'rb') as f:
            [title_embeddings, synopsis_embeddings] = pickle.load(f)

        for i, record in enumerate(RecordStore.test_records):
            record.title_embeddings_w2v = title_embeddings[i]
            record.synopsis_embeddings_w2v = synopsis_embeddings[i]


        
    

