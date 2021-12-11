from typing import NamedTuple, Any, List, Dict
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

from iml_group_proj.features.common.config import RANDOM_STATE

class TrainedModel(NamedTuple):
    name: str
    model: Any # Any functoin that has .predict_proba()
    info: Dict[str, Any] # Information about the training (e.g., is_train?, with_sypnosis?)
        

def train_models(models, X_train, y_train, info: Dict[str, Any] = {}) -> List[TrainedModel]:
    clfs = []   
    for (model, params, name) in models:
        if params is not None:
            clf = HalvingGridSearchCV(model, params, random_state=RANDOM_STATE).fit(X_train, y_train)
            info["best_params"] = str(clf.best_params_)
        else:
            clf = model.fit(X_train, y_train)
            info["best_params"] = "{}"

        clfs.append(TrainedModel(name, clf, info))

    return clfs
