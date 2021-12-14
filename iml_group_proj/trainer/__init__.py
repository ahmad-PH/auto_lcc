import os
import pickle
from typing import NamedTuple, Any, List, Dict
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

from iml_group_proj.config import RANDOM_STATE

class TrainedModel(NamedTuple):
    name: str
    model: Any # Any functoin that has .predict_proba()
    info: Dict[str, Any] # Information about the training (e.g., is_train?, with_sypnosis?)
        

def train_model(model_tuple, X_train, y_train, info: Dict[str, Any] = {}, cache_filename: str = None) -> TrainedModel:
    model, params, name = model_tuple
    
    maybe_model = maybe_load_model(cache_filename)

    if maybe_model:
        print(f"Model found at {cache_filename}, skip training flow...")
        return maybe_model

    print(f"Model not found at {cache_filename}, training {name}")
    if params is not None:
        clf = HalvingGridSearchCV(model, params, random_state=RANDOM_STATE).fit(X_train, y_train)
        info["best_params"] = str(clf.best_params_)
    else:
        clf = model.fit(X_train, y_train)
        info["best_params"] = "{}"
    
    print(f"Finished training!")
    trained_model = TrainedModel(name, model, info)
    if cache_filename is not None:
        print(f"Saving the model to {cache_filename}...")
        save_model(cache_filename, trained_model)
    return TrainedModel(name, clf, info)


def save_model(model_file_name, trained_model):
    with open(model_file_name, "wb") as f:
        pickle.dump(trained_model, f)
    return True

def maybe_load_model(model_file_name) -> TrainedModel:
    if not os.path.isfile(model_file_name):
        return False
    
    with open(model_file_name, "rb") as f:
        return pickle.load(f) # Model object
