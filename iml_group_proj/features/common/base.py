import pandas as pd
from enum import Enum
from typing import Callable, List, Set, Union, Tuple, Any
 

class CleanerId(Enum):
    # Number has no meanings. Could have made string but lazy to type :| 
    TO_PD = 0
    LOWER = 1
    REMOVE_PUNC = 2
    CLEAN_SPACE = 3
    TOKENIZED = 10
    REMOVE_EMPTY_TOKEN = 11
    PT_TOKENIZED = 12
    PT_MODEL = 15
    HASHTAG = 20
    TRUNCATE_WORDS = 21

    STOP_WORD = 25
    CONCAT = 30

    TO_BERT = 40

    NONE = -1 # Don't specify any state to the next function

FrameOrSeries = Union[pd.Series, pd.DataFrame]

# Interface (?)
def prep_pipeline(cleanerId: CleanerId, prereqs: List[CleanerId] = []):

    def decorator(f: Callable):

        def f_applier(
                text_column: Any,
                status: Set[CleanerId] = set()
                ) -> Tuple[FrameOrSeries, Set[CleanerId]]:
            not_done_prereqs = [r for r in prereqs if r not in status]
            assert len(not_done_prereqs) == 0, f"Prerequisites failed, {not_done_prereqs} not done yet."

            new_status = status.copy()
            new_status.add(cleanerId)
            return f(text_column), new_status

        return f_applier

    return decorator

def clean_text(text_column: pd.Series, pipelines: List[Callable]) -> pd.Series:

    status = set()
    # Immutable just to keep our lives sane, might change latter forperformance
    _text_column = text_column.copy() 

    for pipeline_f in pipelines:
        _text_column, status = pipeline_f(_text_column, status)

    return _text_column
