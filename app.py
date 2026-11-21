"""Streamlit app for the Student Performance Predictor. Run:  streamlit run app.py"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

import generate_data
import predictor as sp

st.set_page_config(page_title="Student Performance Predictor", layout="wide")
sns.set_theme(style="whitegrid")
DATA = Path(__file__).parent / "data" / "student_data.csv"


@st.cache_data
def load():
    if not DATA.exists():
        DATA.parent.mkdir(exist_ok=True)
        generate_data.build().to_csv(DATA, index=False)
    return pd.read_csv(DATA)


@st.cache_resource
def train(df):
    return sp.train_all(df)


df = load()
models, table, preds, X_train, X_test, y_train, y_test = train(df)

st.title("Student Performance Predictor")
tab1, tab2, tab3 = st.tabs(["Predict a student", "Model results", "Data exploration"])

with tab1:
    st.write("Enter a student's details to predict the final marks.")
    c1, c2 = st.columns(2)
    study = c1.slider("Study hours per day", 0.0, 12.0, 5.0, 0.5)
    attendance = c1.slider("Attendance (%)", 30.0, 100.0, 80.0, 1.0)
    previous = c1.slider("Previous score", 0.0, 100.0, 60.0, 1.0)
    assign = c2.slider("Assignments score", 0.0, 100.0, 65.0, 1.0)
    gender = c2.selectbox("Gender", ["Male", "Female"])
    parent = c2.selectbox("Parent education", ["High School", "Bachelor", "Master", "PhD"])
    model_name = c2.selectbox("Model", list(models), index=list(models).index(table.index[0]))
    row = pd.DataFrame([{"Study Hours": study, "Attendance": attendance, "Previous Score": previous,
                         "Assignments Score": assign, "Gender": gender, "Parent Education": parent}])
    pred = float(models[model_name].predict(row)[0])
    pred = max(0.0, min(100.0, pred))
    st.metric("Predicted final marks", f"{pred:.1f} / 100")
    label = sp.risk_label(pred)
    (st.error if label.startswith("At risk") else st.success if label == "Good" else st.info)(label)

with tab2:
    st.dataframe(table.round(3), width="stretch")
    fig, ax = plt.subplots(figsize=(6, 5))
    best = table.index[0]
    ax.scatter(y_test, preds[best], alpha=0.4, s=16); ax.plot([0, 100], [0, 100], "r--")
    ax.set(title=f"{best}: actual vs predicted", xlabel="Actual", ylabel="Predicted")
    st.pyplot(fig)

with tab3:
    st.dataframe(df.head(30), width="stretch")
    a, b = st.columns(2)
    fig, ax = plt.subplots(figsize=(5.5, 4)); sns.regplot(data=df, x="Study Hours", y=sp.TARGET, ax=ax,
                                                          scatter_kws={"alpha": .3, "s": 14}, line_kws={"color": "red"})
    a.pyplot(fig)
    fig, ax = plt.subplots(figsize=(5.5, 4)); sns.regplot(data=df, x="Attendance", y=sp.TARGET, ax=ax,
                                                          scatter_kws={"alpha": .3, "s": 14}, line_kws={"color": "red"})
    b.pyplot(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); sns.histplot(df[sp.TARGET], kde=True, ax=ax)
    st.pyplot(fig)
