import functools
import pandas as pd
from typing import Callable, Dict, List, Any
from iml_group_proj.model.training_flow import TrainedModel
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def precision_recall_f1(label, prediction) -> Dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(label, prediction, average="macro")

    return {
            "precision": p,
            "recall": r,
            "f1_score": f
    }

def accuracy(label, prediction) -> Dict[str, float]:
    return {
            "accuracy": accuracy_score(label, prediction)
            }

def evaluate(
        prediction,
        label,
        name: str,
        info: Dict[str, Any],
        flows: List[Callable]
        ):

    flows = flows + [lambda a, b: {"name": name}]
    results = [f(label, prediction) for f in flows] + [info]

    return functools.reduce(
        lambda a,b: {**a, **b}, # Adding two dicts together.
        results
        )

def evaluate_many(
        models: List[TrainedModel],
        X_train,
        y_train,
        X_test,
        y_test,
        evaluation_functions: List[Callable] = [accuracy, precision_recall_f1]
        ) -> List[Dict[str, Any]]:
    results = []
    for trained_model in models:
        clf = trained_model.model
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        train_row = evaluate(y_train_pred, y_train, trained_model.name, trained_model.info, evaluation_functions)
        train_row["is_train"] = True
        results.append(train_row)

        test_row = evaluate(y_test_pred, y_test, trained_model.name, trained_model.info, evaluation_functions)
        test_row["is_train"] = False
        results.append(test_row)

    return results
    
