"""TF-IDF + Logistic Regression and TF-IDF + LightGBM text classifiers."""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from jev_bench.models.base import Model

LR_C_GRID = (0.25, 1.0, 4.0, 16.0)
LGBM_MAX_FEATURES = 20_000
LGBM_EARLY_STOPPING_ROUNDS = 50


def _lgbm_params(seed: int, n_labels: int) -> dict:
    return {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.5,
        "random_state": seed,
        "objective": "binary" if n_labels == 2 else "multiclass",
        "verbose": -1,
    }


def fit_lgbm(x_train, y_train, x_val, y_val, seed: int, n_labels: int, **fit_kwargs) -> lgb.LGBMClassifier:
    clf = lgb.LGBMClassifier(**_lgbm_params(seed, n_labels))
    clf.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=False)],
        **fit_kwargs,
    )
    return clf


class TfidfLR(Model):
    name = "tfidf_lr"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        x_train = self.vec.fit_transform(train["text"])
        x_val = self.vec.transform(val["text"])
        best = None
        for c in LR_C_GRID:
            clf = LogisticRegression(C=c, max_iter=2000, random_state=self.seed)
            clf.fit(x_train, train["label"])
            acc = clf.score(x_val, val["label"])
            if best is None or acc > best[0]:
                best = (acc, c, clf)
        self.val_accuracy, self.c, self.clf = best

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self.vec.transform(df["text"]))

    def run_info(self) -> dict:
        return {"C": self.c, "val_accuracy": self.val_accuracy}


class TfidfLGBM(Model):
    name = "tfidf_lgbm"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=LGBM_MAX_FEATURES)
        x_train = self.vec.fit_transform(train["text"]).astype(np.float32)
        x_val = self.vec.transform(val["text"]).astype(np.float32)
        self.clf = fit_lgbm(x_train, train["label"], x_val, val["label"], self.seed, len(self.task.labels))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self.vec.transform(df["text"]).astype(np.float32))

    def run_info(self) -> dict:
        return {"best_iteration": int(self.clf.best_iteration_ or 0)}
