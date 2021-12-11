from sklearn.neural_network import MLPClassifier
from iml_group_proj.features.common.config import RANDOM_STATE

BASIC_MLP = MLPClassifier(
        random_state=RANDOM_STATE,
        max_iter=100,
        hidden_layer_sizes=(100, 100),
        early_stopping=True
        )
