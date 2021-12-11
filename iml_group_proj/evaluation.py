import functools
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Any
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from iml_group_proj.features.common.config import RANDOM_STATE
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

from iml_group_proj.train_models import TrainedModel

def evaluate_many(models: List[TrainedModel], X_train, y_train, X_test, y_test):
    results = []
    for trained_model in models:
        clf = trained_model.model
        y_train_pred = clf.predict_proba(X_train)
        y_test_pred = clf.predict_proba(X_test)

        train_row = evaluate(y_train_pred, y_train, trained_model.name, trained_model.info)
        train_row["is_train"] = True
        results.append(train_row)

        test_row = evaluate(y_test_pred, y_test, trained_model.name, trained_model.info)
        test_row["is_train"] = False
        results.append(test_row)

    return pd.DataFrame(results)
    
def precision_recall_f1(label, prediction) -> Dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(label, np.argmax(prediction, axis=1), average="macro")

    return {
            "precision": p,
            "recall": r,
            "f1_score": f
    }

def accuracy(label, prediction) -> Dict[str, float]:
    return {
            "accuracuy": accuracy_score(label, np.argmax(prediction, axis=1))
            }

def evaluate(
        prediction,
        label,
        name: str,
        info: Dict[str, Any],
        flows: List[Callable] = [accuracy, precision_recall_f1]
        ):

    flows = flows + [lambda a, b: {"name": name}]
    results = [f(label, prediction) for f in flows] + [info]

    return functools.reduce(
        lambda a,b: {**a, **b}, # Adding two dicts together.
        results
        )


