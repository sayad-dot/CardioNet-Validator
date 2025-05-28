# src/preprocessing/column_mapping.py

UNIFIED_COLUMNS = {
    # Common mapping across all datasets
    "age": "age",
    "sex": "sex",
    "sex ": "sex",  # for dataset1 with space
    "cp": "chest pain type",
    "chest pain type": "chest pain type",
    "trestbps": "resting blood pressure",
    "resting blood pressure": "resting blood pressure",
    "chol": "serum cholestoral",
    "serum cholestoral": "serum cholestoral",
    "fbs": "fasting blood sugar",
    "fasting blood sugar": "fasting blood sugar",
    "restecg": "resting electrocardiographic results",
    "resting electrocardiographic results": "resting electrocardiographic results",
    "thalach": "max heart rate",
    "max heart rate": "max heart rate",
    "exang": "exercise induced angina",
    "exercise induced angina": "exercise induced angina",
    "oldpeak": "oldpeak",
    "slope": "ST segment",
    "ST segment": "ST segment",
    "ca": "major vessels",
    "major vessels": "major vessels",
    "thal": "thal",
    "target": "heart disease",
    "heart disease": "heart disease"
}
