"""Student Performance Predictor: full pipeline. Run:  python main.py"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib

import predictor as sp

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "student_data.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

# %% 1. Data collection
if not DATA.exists():
    import generate_data
    DATA.parent.mkdir(exist_ok=True)
    generate_data.build().to_csv(DATA, index=False)
df = pd.read_csv(DATA)
print(f"Loaded {len(df)} students. Columns: {list(df.columns)}")

# %% 2. Preprocessing (imputing, encoding and scaling happen inside the model pipelines)
print("\nMissing values per column:")
print(df.isna().sum()[df.isna().sum() > 0].to_string())
print("\nSummary statistics:")
print(df.describe().round(2).to_string())

# %% 3. EDA
num_cols = df.select_dtypes("number").columns
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation matrix")
fig.tight_layout(); fig.savefig(OUT / "01_correlation_heatmap.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.regplot(data=df, x="Study Hours", y=sp.TARGET, scatter_kws={"alpha": 0.35, "s": 18}, line_kws={"color": "red"}, ax=ax)
ax.set_title(f"Study hours vs final marks (r = {df['Study Hours'].corr(df[sp.TARGET]):.2f})")
fig.tight_layout(); fig.savefig(OUT / "02_study_hours_vs_marks.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.regplot(data=df, x="Attendance", y=sp.TARGET, scatter_kws={"alpha": 0.35, "s": 18}, line_kws={"color": "red"}, ax=ax)
ax.set_title(f"Attendance vs final marks (r = {df['Attendance'].corr(df[sp.TARGET]):.2f})")
fig.tight_layout(); fig.savefig(OUT / "03_attendance_vs_marks.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.histplot(df[sp.TARGET], bins=25, kde=True, ax=ax)
ax.axvline(sp.PASS_MARK, color="red", ls="--", label=f"Pass mark ({sp.PASS_MARK})")
ax.set_title("Distribution of final marks"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "04_marks_distribution.png", dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.barplot(data=df, x="Parent Education", y=sp.TARGET, order=["High School", "Bachelor", "Master", "PhD"],
            ax=axes[0], color="#4c72b0")
axes[0].set_title("Average marks by parent education")
sns.barplot(data=df, x="Gender", y=sp.TARGET, ax=axes[1], color="#55a868")
axes[1].set_title("Average marks by gender")
fig.tight_layout(); fig.savefig(OUT / "05_group_comparison.png", dpi=130); plt.close(fig)

# %% 4-6. Model building, training, testing, evaluation
models, table, preds, X_train, X_test, y_train, y_test = sp.train_all(df)
print(f"\nTrain size: {len(X_train)}, test size: {len(X_test)}")
print("\nModel comparison on unseen test data:")
print(table.round(3).to_string())
table.round(4).to_csv(OUT / "model_metrics.csv")

best = table.index[0]
print(f"\nBest model: {best}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
for ax, (name, p) in zip(axes, preds.items()):
    ax.scatter(y_test, p, alpha=0.4, s=16)
    ax.plot([0, 100], [0, 100], "r--")
    ax.set(title=f"{name} (R2 = {table.loc[name, 'R2']:.3f})", xlabel="Actual marks")
axes[0].set_ylabel("Predicted marks")
fig.tight_layout(); fig.savefig(OUT / "06_actual_vs_predicted.png", dpi=130); plt.close(fig)

compare = pd.DataFrame({"Actual": y_test.values, "Predicted": preds[best].round(1)})
compare["Error"] = (compare["Predicted"] - compare["Actual"]).round(1)
compare.to_csv(OUT / "actual_vs_predicted.csv", index=False)
print("\nFirst 8 test students (best model):")
print(compare.head(8).to_string(index=False))

rf = models["Random Forest"]
imp = pd.Series(rf.named_steps["model"].feature_importances_, index=sp.feature_names(rf)).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
imp.plot.barh(ax=ax, color="#4c72b0"); ax.set_title("Random Forest feature importance")
fig.tight_layout(); fig.savefig(OUT / "07_feature_importance.png", dpi=130); plt.close(fig)

# %% Save the best model for the web app
joblib.dump(models[best], OUT / "best_model.joblib")
print(f"\nSaved best model to {OUT / 'best_model.joblib'}")
