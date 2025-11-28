import os
import mlflow
import pandas as pd
from dotenv import load_dotenv

load_dotenv() # to track uri


def to_pandas(data: dict):
    X = pd.DataFrame([data])
    X["Sex"] = (X["Sex"] == "male").astype(int)
    X[["Pclass", "SibSp", "Parch", "Age"]] = X[["Pclass", "SibSp", "Parch", "Age"]].astype(int)
    return X


def get_results(data: dict):
    X = to_pandas(data)
    model = mlflow.sklearn.load_model("models:/titanic-RF/Production")
    pred = int(model.predict(X)[0])
    prob = model.predict_proba(X)[:, 1]
    factors = None

    return pred, prob, factors
