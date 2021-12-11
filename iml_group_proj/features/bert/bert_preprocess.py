import torch
from torch import Tensor
from typing import Sequence
from transformers import BertModel, BertTokenizer
from iml_group_proj.features.common.base import CleanerId, prep_pipeline

# BERT_MODEL_NAME = "bert-base-uncased"
BERT_MODEL_NAME = "prajjwal1/bert-tiny"
# BERT_MODEL_NAME = "prajjwal1/bert-small" # Twice as big as the tiny one
MAX_TOKENS_LENGTH = 512 # Will effect sypnosis! 
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
model = BertModel.from_pretrained(BERT_MODEL_NAME)

@prep_pipeline(CleanerId.TO_BERT)
def get_bert_embeddings(sentences: Sequence[str]):
    """
    Return a list of Tensors.
    """

    outputs = []
    
    for sentence in sentences:
        inputs = bert_tokenizer(sentence, truncation=True, max_length=MAX_TOKENS_LENGTH, return_tensors="pt")

        output = model(**inputs)
        outputs.append(output.last_hidden_state)


    return outputs

@prep_pipeline(CleanerId.NONE, [CleanerId.TO_BERT])
def combine_embeddings(embeddings: Sequence[Tensor]):
    """
    Average the whole tensors. Make sure the embeddings aren't padded
    """
    return torch.stack([torch.sum(bert_emb, axis=1).squeeze(0) for bert_emb in embeddings], axis=0)
