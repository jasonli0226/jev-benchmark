"""Native-feature baselines for Adult (Track B reference point)."""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from jev_bench.data.sources import ADULT_CATEGORICAL, ADULT_FEATURES, ADULT_NUMERIC
from jev_bench.models.base import Model
from jev_bench.models.classical import LR_C_GRID, fit_lgbm


class TabularLR(Model):
    name = "tabular_lr"
    input_kind = "tabular"

    def _pipeline(self, c: float):
        pre = ColumnTransformer(
            [
                ("num", StandardScaler(), ADULT_NUMERIC),
                ("cat", OneHotEncoder(handle_unknown="ignore"), ADULT_CATEGORICAL),
            ]
        )
        return make_pipeline(pre, LogisticRegression(C=c, max_iter=2000, random_state=self.seed))

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        best = None
        for c in LR_C_GRID:
            pipe = self._pipeline(c).fit(train[ADULT_FEATURES], train["label"])
            acc = pipe.score(val[ADULT_FEATURES], val["label"])
            if best is None or acc > best[0]:
                best = (acc, c, pipe)
        self.val_accuracy, self.c, self.pipe = best

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(df[ADULT_FEATURES])

    def run_info(self) -> dict:
        return {"C": self.c, "val_accuracy": self.val_accuracy}


class TabularLGBM(Model):
    name = "tabular_lgbm"
    input_kind = "tabular"

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[ADULT_FEATURES].copy()
        for col in ADULT_CATEGORICAL:
            x[col] = pd.Categorical(x[col], categories=self.categories[col])
        return x

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        self.categories = {col: sorted(train[col].unique()) for col in ADULT_CATEGORICAL}
        self.clf = fit_lgbm(
            self._frame(train), train["label"], self._frame(val), val["label"], self.seed, len(self.task.labels)
        )

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self._frame(df))

    def run_info(self) -> dict:
        return {"best_iteration": int(self.clf.best_iteration_ or 0)}
