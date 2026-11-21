# Student performance predictor

Sqrock IT Solutions, Data Science internship, Phase 1, Task 2.

This project predicts a student's final marks from study hours, attendance, previous score, assignment score, gender and parent education. It also labels students who are predicted to fall below the pass mark (50) so they can get extra support.

## Run it

```bash
pip install -r requirements.txt
python generate_data.py     # optional
python main.py              # pipeline, charts and metrics in outputs/
streamlit run app.py        # web app: enter a student, get a prediction
```

## Dataset

`data/student_data.csv` is synthetic (1000 students). Columns: Study Hours, Attendance, Previous Score, Assignments Score, Gender, Parent Education, Final Marks (target). A few values are missing on purpose. You can swap in a Kaggle student dataset if you rename the target column to `Final Marks`.

## What each step does

1. Data collection: reads the CSV.
2. Preprocessing: median imputation for numbers, most frequent for categories, one-hot encoding, scaling for Linear Regression, and an 80/20 train/test split. All of this sits inside scikit-learn pipelines, so nothing leaks from the test set.
3. EDA: correlation heatmap, study hours vs marks, attendance vs marks, marks histogram, averages by parent education and gender.
4. Models: Linear Regression, Decision Tree Regressor, Random Forest Regressor.
5. Training and testing: fit on 800 students, test on 200 unseen ones, compare predicted and actual marks.
6. Evaluation: MAE, MSE and R2.

## Results on the test set

| Model | MAE | MSE | R2 |
|---|---|---|---|
| Linear Regression | 3.149 | 15.417 | 0.808 |
| Random Forest | 3.358 | 17.555 | 0.781 |
| Decision Tree | 4.218 | 27.558 | 0.657 |

The data was generated with mostly linear relationships, so Linear Regression edges out Random Forest. On real data Random Forest is usually the safer pick, which is why both are in the app.

## Files

- `predictor.py`: preprocessing, models, metrics
- `main.py`: full pipeline, writes to `outputs/`
- `app.py`: Streamlit app
- `generate_data.py`: builds the sample dataset

## Deploy

Push to GitHub and create a Streamlit Community Cloud app with `app.py` as the main file.
