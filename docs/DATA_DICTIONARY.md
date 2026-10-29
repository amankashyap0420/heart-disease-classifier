# Data dictionary

Schema of the original `processed.cleveland.data`, not a remapped Kaggle CSV. The raw file has no header; column order is the order below. A question mark denotes a missing value. The loader retains missing features for fold-specific imputation.

| Column | Meaning / encoding |
|---|---|
| age | Age in years |
| sex | Recorded binary sex code: 0 female, 1 male |
| cp | Chest pain: 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic |
| trestbps | Resting blood pressure on hospital admission, mm Hg |
| chol | Serum cholesterol, mg/dL |
| fbs | Fasting blood sugar above 120 mg/dL: 1 yes, 0 no |
| restecg | Resting ECG: 0 normal, 1 ST-T abnormality, 2 probable/definite LV hypertrophy |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina: 1 yes, 0 no |
| oldpeak | ST depression induced by exercise relative to rest |
| slope | Peak exercise ST slope: 1 upsloping, 2 flat, 3 downsloping |
| ca | Count of major vessels colored by fluoroscopy, 0–3 |
| thal | Thallium test: 3 normal, 6 fixed defect, 7 reversible defect |
| num | Original target 0–4; code maps 0 to absence and 1–4 to presence |

`num` is not an input. The derived binary target is not a future-event label. `thal` is the test-result category in this dataset, not a diagnosis of thalassemia.

The application's broad numeric guards prevent clearly invalid inputs; they are not clinical normal ranges. Training minima/maxima are displayed separately. Missing values are supported but extensive imputation reduces the information available to the classifier.

Source: [UCI Heart Disease variable documentation](https://archive.ics.uci.edu/dataset/45/heart+disease). Citation and license are recorded in the README and `data/provenance.json`.
