"""Creates a synthetic student dataset (data/student_data.csv)."""
import numpy as np
import pandas as pd
from pathlib import Path


def build(n=1000, seed=7):
    rng = np.random.default_rng(seed)
    study = np.clip(rng.normal(5, 2, n), 0.5, 12).round(1)
    attendance = np.clip(rng.normal(80, 12, n), 35, 100).round(1)
    previous = np.clip(0.35 * study * 7 + rng.normal(45, 12, n), 20, 100).round(1)
    assignments = np.clip(0.4 * previous + 3 * study + rng.normal(25, 8, n), 10, 100).round(1)
    gender = rng.choice(["Male", "Female"], n)
    parent_edu = rng.choice(["High School", "Bachelor", "Master", "PhD"], n, p=[0.4, 0.35, 0.18, 0.07])
    edu_bonus = pd.Series(parent_edu).map({"High School": 0, "Bachelor": 1.5, "Master": 2.5, "PhD": 3}).values

    final = (0.30 * previous + 0.25 * assignments + 0.18 * attendance + 2.2 * study
             + edu_bonus + rng.normal(0, 4, n) - 10)
    final = np.clip(final, 0, 100).round(1)

    df = pd.DataFrame({
        "Study Hours": study, "Attendance": attendance, "Previous Score": previous,
        "Assignments Score": assignments, "Gender": gender, "Parent Education": parent_edu,
        "Final Marks": final,
    })
    # A few missing values so preprocessing has real work to do
    for col, k in [("Study Hours", 15), ("Attendance", 20), ("Assignments Score", 12), ("Parent Education", 18)]:
        df.loc[rng.choice(n, k, replace=False), col] = np.nan
    return df


if __name__ == "__main__":
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    df = build()
    df.to_csv(out / "student_data.csv", index=False)
    print(f"Saved {len(df)} rows to {out / 'student_data.csv'}")
