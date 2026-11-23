# Two classic leaks in omics machine learning, and the cross-validation that avoids them.
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _model():
    return LogisticRegression(max_iter=2000)


def leaky_selection_cv(X, y, k=20, folds=5, seed=0):
    # WRONG: pick the k best genes using ALL samples, then cross-validate on them.
    X_sel = SelectKBest(f_classif, k=k).fit_transform(X, y)
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return cross_val_score(make_pipeline(StandardScaler(), _model()), X_sel, y, cv=cv).mean()


def honest_selection_cv(X, y, k=20, folds=5, seed=0):
    # RIGHT: selection happens inside each training fold, as part of the pipeline.
    pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=k), _model())
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return cross_val_score(pipe, X, y, cv=cv).mean()


def sample_level_cv(X, y, folds=5, seed=0):
    # WRONG when several samples come from one patient: the same patient lands in train and test.
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return cross_val_score(make_pipeline(StandardScaler(), _model()), X, y, cv=cv).mean()


def patient_level_cv(X, y, patients, folds=5):
    # RIGHT: every sample from a patient stays in the same fold.
    return cross_val_score(make_pipeline(StandardScaler(), _model()), X, y,
                           cv=GroupKFold(folds), groups=patients).mean()
