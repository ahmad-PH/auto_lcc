from typing import *


class Record:
    def __init__(self):
        self.id: int = None
        self.cls: str = None
        self.title: str = None
        self.synopsis: str = None
            
        self.title_embeddings_bert: List[Any] = None
        self.synopsis_embeddings_bert: List[Any] = None

        self.title_embeddings_w2v: List[Any] = None
        self.synopsis_embeddings_w2v: List[Any] = None
        
class RecordStore:
    def __init__(self) -> None:
        pass

    def load_bert(self):
        pass # TODO

    def load_w2v(self):
        pass # TODO

    def get_record(i: int) -> Record:
        pass 
