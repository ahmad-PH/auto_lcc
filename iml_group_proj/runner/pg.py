import time
from iml_group_proj.config import RANDOM_STATE
from iml_group_proj.evaluation import evaluate_many
from iml_group_proj.data.utils import load_libofc_df
from iml_group_proj.data.preprocess import AverageEmbeddingsPerRecord
from iml_group_proj.data.embeddings import EmbeddingsDataLoader, EmbeddingsType
from iml_group_proj_back.train_models import train_models

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier as OVC

DATA_PATH = "github_data"
start = time.time()
# List of Embeddings or PCA-ed tf-idf. 
embeddings_type = EmbeddingsType.TFIDF
data = EmbeddingsDataLoader.load(DATA_PATH, embeddings_type)
X_train, y_train, X_test, y_test = AverageEmbeddingsPerRecord.prep(data, embeddings_type)

# DataFeeder (Averaging list of embeddings) or just return the whole thing.
models = [
        (MLPClassifier(random_state=1, max_iter=100), None, 'MLP_100'),
        (SVC(C=100, kernel="rbf", gamma=0.001, random_state=RANDOM_STATE), None, "SVC"),
        (OVC(SVC(C=100, kernel="rbf", gamma=0.001, random_state=RANDOM_STATE)), None, "OVC-SVC"),
        (GaussianNB(), None,'NaiveBayes'),
        ]

trained_models = train_models(models, X_train, y_train, {"embeddings_type": embeddings_type})
result_df = evaluate_many(trained_models, X_train, y_train, X_test, y_test)

print("Training done!, took {time.time() - start:.2f}s")
print(result_df.head(40))
