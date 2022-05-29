import os
import pickle
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from preprocessing import clean, clean_df, stop_words
from typehints import *

dirname = os.path.dirname(__file__)
SAVE_FILE = os.path.join(dirname, 'saves/model_manager.pkl')


class SvmManager:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)

        self.pipe = Pipeline([('vectorizer', self.vectorizer),
                              ('svc', SVC(cache_size=1000, probability=True))])

        print(self.pipe.get_params().keys())

        self.param_grid = [{'svc__kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
                            'svc__C': (1, 10, 100, 1000)}]
        self.model: SVC = SVC()
        self.acc = 0

    def train(self, data: List[TrainData]) -> TrainInfo:
        df = clean_df(pd.DataFrame(data))

        X_train, X_test, y_train, y_test = split_data(df)

        self.model = HalvingGridSearchCV(
            estimator=self.pipe,
            param_grid=self.param_grid,
            n_jobs=-1,
            scoring="accuracy",
            factor=2,
            min_resources=10
        )
        self.model.fit(X_train, y_train)
        self.acc = self.model.score(X_test, y_test)

        save(self)
        return {'accuracy': self.acc}

    def predict(self, data: PredictData) -> Prediction:
        x = [clean(data['text'])]
        prediction = self.model.predict(x)

        probas = self.model.predict_proba(x)[0]  # [0] -> get the first (only) sample
        confidence = probas[np.argmax(probas)]

        return {
            'id': data['id'],
            'label': int(prediction[0]),
            'confidence': int(confidence * 100)
        }

    def info(self) -> dict:
        return {'model': {'accuracy': self.acc, 'name': str(self.pipe['svc']), 'params': str(self.pipe['svc'].get_params())}}


def split_data(data: pd.DataFrame):
    y = data.label.values
    X_train, X_test, y_train, y_test = train_test_split(
        data.clean.values, y,
        test_size=0.3,
        shuffle=True,
        stratify=y  # (for y imbalances) https://scikit-learn.org/stable/modules/cross_validation.html#stratification
    )
    return X_test, X_train, y_test, y_train


def save(svm_mngr: SvmManager):
    with open(SAVE_FILE, 'wb') as f:
        pickle.dump(svm_mngr, f)


def load_SvmManager() -> SvmManager:
    with open(SAVE_FILE, 'rb') as f:
        try:
            return pickle.load(f)
        except EOFError:
            return SvmManager()
