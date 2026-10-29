from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "processed.cleveland.data"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
SEED = 42
TEST_SIZE = 0.20
FOLDS = 5
THRESHOLD = 0.50
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORIES = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"},
    "fbs": {0: "At most 120 mg/dL", 1: "Above 120 mg/dL"},
    "restecg": {0: "Normal", 1: "ST-T wave abnormality", 2: "Probable / definite LV hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}
# Broad input guards, not clinical normal ranges. Training ranges are shown separately.
BOUNDS = {"age": (1, 120), "trestbps": (1, 300), "chol": (1, 1000),
          "thalach": (1, 300), "oldpeak": (-10, 20), "ca": (0, 3)}
LABELS = {"age": "Age (years)", "sex": "Sex (dataset encoding)", "cp": "Chest pain category",
          "trestbps": "Resting blood pressure (mm Hg)", "chol": "Serum cholesterol (mg/dL)",
          "fbs": "Fasting blood sugar", "restecg": "Resting ECG", "thalach": "Maximum heart rate (bpm)",
          "exang": "Exercise-induced angina", "oldpeak": "Exercise ST depression (oldpeak)",
          "slope": "Peak exercise ST slope", "ca": "Major vessels visible on fluoroscopy",
          "thal": "Thallium test result"}
