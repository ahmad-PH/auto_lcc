import re
import nltk
from nltk.corpus import stopwords
import string
from typing import List
from iml_group_proj.features.common.base import prep_pipeline, CleanerId, FrameOrSeries
 
stop_words = set(stopwords.words('english'))

def sample_data(sample_frac: float):
    @prep_pipeline(CleanerId.LOWER)
    def f(text_col: FrameOrSeries):
        return text_col.sample(frac=sample_frac)
    return f


@prep_pipeline(CleanerId.LOWER)
def lower_case(text_col: FrameOrSeries):
    return text_col.apply(lambda s: s.lower())

@prep_pipeline(CleanerId.REMOVE_PUNC)
def remove_punc(text_col: FrameOrSeries):
    remover = lambda s: s.translate(str.maketrans('', '', string.punctuation))
    return text_col.apply(remover)

@prep_pipeline(CleanerId.CLEAN_SPACE)
def clean_space(text_col: FrameOrSeries):
    remover = lambda s: re.sub(" +", ' ', s)
    return text_col.apply(remover)

@prep_pipeline(CleanerId.TOKENIZED)
def tokenize_words(text_col: FrameOrSeries):
    return text_col.apply(lambda s: s.split(" "))

@prep_pipeline(CleanerId.STOP_WORD, [CleanerId.TOKENIZED])
def remove_stop_words(text_col: FrameOrSeries):
    return text_col.apply(lambda tokens: [token for token in tokens if token not in stop_words])

@prep_pipeline(CleanerId.REMOVE_EMPTY_TOKEN, [CleanerId.TOKENIZED])
def remove_empty_token(text_col: FrameOrSeries):
    return text_col.apply(lambda tokens: [token for token in tokens if token != ''])

@prep_pipeline(CleanerId.HASHTAG, [CleanerId.TOKENIZED])
def remove_hashtag(text_col: FrameOrSeries):

    def filtering(tokens: List[str]):
        return [word for word in tokens if "@" not in word]

    return text_col.apply(filtering)

def truncate_words(max_words: int):
    @prep_pipeline(CleanerId.TRUNCATE_WORDS, [CleanerId.TOKENIZED])
    def f(text_col: FrameOrSeries):

        def filtering(tokens: List[str]):
            return tokens[:max_words]

        return text_col.apply(filtering)
    return f

@prep_pipeline(CleanerId.REMOVE_EMPTY_TOKEN, [CleanerId.TOKENIZED])
def concat_tokens(text_col: FrameOrSeries):
    return text_col.apply(lambda tokens: " ".join(tokens))
