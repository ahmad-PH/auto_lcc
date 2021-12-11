import pytest

from data import RecordStore

def test_recordstore_load():
    assert len(RecordStore.train_records) == 0
    RecordStore.load()
    assert RecordStore.train_records[0] != []

def test_recordstore_w2v():
    print("Loading RecordStore")
    RecordStore.load()
    print("Loading w2v embeddings")
    RecordStore.load_w2v()
    assert RecordStore.train_records[0].title_embeddings_w2v is not None

def test_recordstore_bert():
    print("Loading RecordStore")
    RecordStore.load()
    print("Loading bert embeddings")
    RecordStore.load_bert()
    assert RecordStore.train_records[0].title_embeddings_bert is not None

