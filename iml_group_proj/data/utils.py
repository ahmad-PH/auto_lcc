import pickle
import pandas as pd
from typing import List, Tuple

def load_libofc_df(data_path):
    def tuple_to_df(data: List[Tuple]) -> pd.DataFrame:
        return pd.DataFrame(data, columns=["class", "title", "synopsis", "id"])

    with open(data_path, 'rb') as f:
        classes = pickle.load(f)
        train = pickle.load(f)
        test = pickle.load(f)

    return classes, tuple_to_df(train), tuple_to_df(test)
