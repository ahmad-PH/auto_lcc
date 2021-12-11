import pickle
import pandas as pd
from os import PathLike
from typing import Tuple, List

from iml_group_proj.features.common.config import LOC_DATA_PATH

def load(data_path: PathLike[str] = LOC_DATA_PATH, sample_frac:float = 1.0) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    with open(data_path, 'rb') as f:
        classes = pickle.load(f)
        train = pickle.load(f)
        test = pickle.load(f)

        return classes, tuple_to_df(train).sample(frac=sample_frac), tuple_to_df(test).sample(frac=sample_frac)


def tuple_to_df(data: List[Tuple]) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=["class", "title", "sypnosis", "id"])
    df["class_name"] = df["class"].apply(lambda x: class_to_name_map.get(x, "Unknown"))
    return df

class_to_name_map = {
        "A":	"General Works",
        "B":	"Philosophy. Psychology. Religion",
        "C":	"Auxiliary Sciences of History",
        "D":	"World History and History of Europe, Asia, Africa, Australia, New Zealand, etc.",
        "E":	"History of the Americas (E)",
        "F":	"History of the Americas (F)",
        "G":	"Geography. Anthropology. Recreation",
        "H":	"Social Sciences",
        "J":	"Political Science",
        "K":	"Law",
        "L":	"Education",
        "M":	"Music and Books on Music",
        "N":	"Fine Arts",
        "P":	"Language and Literature",
        "Q":	"Science",
        "R":	"Medicine",
        "S":	"Agriculture",
        "T":	"Technology",
        "U":	"Military Science",
        "V":	"Naval Science",
        "Z":	"Bibliography. Library Science. Information Resources (General)",
        }
