import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def predict_unknown(X_value, model_path="our_model.pkl"):
    loaded_model = pickle.load(open(model_path, "rb"))

    y_unknown = loaded_model.predict(np.array([X_value]).reshape(-1, 1))

    print("The question: X =", X_value)
    print("The answer: y =", y_unknown[0])
    print("... Double check...")
    print("y = a*x + b")
    print("a =", loaded_model.coef_[0])
    print("b =", loaded_model.intercept_[0])
    print("y = a*x + b")

    predicted_value = loaded_model.coef_[0] * X_value + loaded_model.intercept_[0]
    print(predicted_value)
    return y_unknown[0]


def train_and_export_model(X, y, path_to_csv="data/10_points.csv", model_path="our_model.pkl"):
    try:
        df = pd.read_csv(path_to_csv)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["x", "y"])

    new_data = pd.DataFrame({"x": [X], "y": [y]})
    df = df.append(new_data, ignore_index=True)

    df.to_csv(path_to_csv, index=False)

    df = pd.read_csv(path_to_csv)
    X_data = df["x"].values.reshape(-1, 1)
    y_data = df["y"].values.reshape(-1, 1)

    try:
        our_model = pickle.load(open(model_path, "rb"))
    except FileNotFoundError:
        our_model = LinearRegression()

    our_model.fit(X_data, y_data)

    print("y = a*x + b")
    print("a =", our_model.coef_[0][0])
    print("b =", our_model.intercept_[0])

    print("... Eksport modelu...")
    pickle.dump(our_model, open(model_path, "wb"))
