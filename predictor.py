"""Core logic for the Student Performance Predictor (used by main.py and app.py)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

TARGET = "Final Marks"
PASS_MARK = 50


def split_columns(df):
    X = df.drop(columns=[TARGET])
    cat = X.select_dtypes(include="object").columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return num, cat


def make_preprocessor(num, cat, scale=True):
    """Impute missing values, encode categories, and (optionally) scale numeric columns."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    parts = [("num", Pipeline(num_steps), num)]
    if cat:
        parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat))
    return ColumnTransformer(parts)


def make_models(num, cat):
    return {
        "Linear Regression": Pipeline([("prep", make_preprocessor(num, cat, True)),
                                       ("model", LinearRegression())]),
        "Decision Tree": Pipeline([("prep", make_preprocessor(num, cat, False)),
                                   ("model", DecisionTreeRegressor(max_depth=6, min_samples_leaf=8,
                                                                   random_state=42))]),
        "Random Forest": Pipeline([("prep", make_preprocessor(num, cat, False)),
                                   ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                                                   random_state=42, n_jobs=-1))]),
    }


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse, "R2": r2_score(y_true, y_pred)}


def train_all(df, test_size=0.2, seed=42):
    df = df.dropna(subset=[TARGET])
    num, cat = split_columns(df)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    models, results, preds = make_models(num, cat), {}, {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds[name] = m.predict(X_test)
        results[name] = evaluate(y_test, preds[name])
    table = pd.DataFrame(results).T.sort_values("R2", ascending=False)
    return models, table, preds, X_train, X_test, y_train, y_test


def risk_label(marks):
    if marks < PASS_MARK:
        return "At risk (needs extra support)"
    if marks < 65:
        return "Average"
    return "Good"


def feature_names(model):
    prep = model.named_steps["prep"]
    return list(prep.get_feature_names_out())
