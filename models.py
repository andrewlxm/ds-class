from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

from flop import (
    flops_knn_per_sample,
    flops_linearsvc_per_sample,
    flops_logreg_per_sample,
    flops_maxabs_scaler,
    flops_rf_per_sample,
    flops_standard_scaler,
)

#Define the models to benchmark
def build_models():

    models = {
        "Random Forest": RandomForestClassifier(min_samples_leaf=1, min_samples_split=2, n_estimators=200),

        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),

        "LinearSVC": LinearSVC(max_iter=5000),

        "StandardScaler + LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000))
        ]),

        "MaxAbsScaler + SVC": Pipeline([
            ("scaler", MaxAbsScaler()),
            ("clf", LinearSVC(max_iter=5000))
        ])
    }

    return models

def estimate_flops(name, model, d, n_train, n_eval):

    if "KNN" in name:
        per_sample = flops_knn_per_sample(model, d)
        return per_sample, per_sample * n_eval

    if "LinearSVC" in name:
        n_classes = len(model.classes_)
        per_sample = flops_linearsvc_per_sample(d, n_classes)
        return per_sample, per_sample * n_eval

    if "StandardScaler" in name:
        n_classes = len(model.named_steps["clf"].classes_)
        per_sample = flops_standard_scaler(d) + flops_logreg_per_sample(d, n_classes)
        return per_sample, per_sample * n_eval

    if "MaxAbsScaler" in name:
        n_classes = len(model.named_steps["clf"].classes_)
        per_sample = flops_maxabs_scaler(d) + flops_linearsvc_per_sample(d, n_classes)
        return per_sample, per_sample * n_eval

    if "Random Forest" in name:
        per_sample = flops_rf_per_sample(model)
        return per_sample, per_sample * n_eval

    return None, None