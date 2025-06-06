CardioNet-Validator 

A robust, end-to-end machine learning pipeline for predicting heart disease from clinical parameters, built with Python and XGBoost. It performs data collection, preprocessing, model training, rigorous cross-dataset validation, and final model integration for inference.

Project Structure

CardioNet-Validator/
├── data/
│   ├── dataset1.csv           # Original Kaggle dataset 1
│   ├── dataset2.csv           # Original Kaggle dataset 2
│   ├── dataset3.csv           # Original Kaggle dataset 3
│   └── processed/
│       ├── clean_dataset1.csv
│       ├── clean_dataset2.csv
│       └── clean_dataset3.csv
├── src/
│   ├── preprocessing/         # Data cleaning & feature engineering
│   ├── models/                # Model utils and per-dataset training scripts
│   ├── validation/            # Cross-dataset validation code
│   └── pipeline/              # Final training & prediction scripts
├── outputs/
│   ├── models_datasetX/       # Serialized models per dataset
│   ├── cross_dataset_validation_summary.csv
│   └── final_model/
│       └── xgb_dataset3.pkl   # Final XGBoost model
└── reports/
    ├── metrics/               # CSVs of evaluation results
    └── figures/               # Confusion matrices & ROC curves

Quick Start

1.Clone & install

git clone https://github.com/yourusername/CardioNet-Validator.git
cd CardioNet-Validator
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

2.Preprocess datasets 1–3 (already committed):

python -m src.preprocessing.preprocess_dataset1
python -m src.preprocessing.preprocess_dataset2
python -m src.preprocessing.preprocess_dataset3


3.Cross-Dataset Validation (Phase 5–6):

python -m src.validation.cross_dataset_validation

(Produces outputs/cross_dataset_validation_summary.csv with accuracy, F1, ROC-AUC for all train→test combos.)

4.Train Final Model (XGBoost on Dataset 3):

python -m src.pipeline.train_final_model

(Saves model to outputs/final_model/xgb_dataset3.pkl)

5.Predict on new data:

python -m src.pipeline.predict_final_model data/processed/clean_dataset3.csv
(Prints input plus predicted_label and predicted_proba)

Key Results

Best generalizing model: XGBoost trained on Dataset 3

Test on Dataset 1 → F1: 0.85, AUC: 0.92

Test on Dataset 2 → F1: 0.74, AUC: 0.89

Cross-dataset evaluation demonstrates robust generalization across different data sources.


Tech Stack

Python 3.12

pandas, numpy

scikit-learn, XGBoost, joblib

matplotlib, seaborn (for plots in reports/)


Next Steps

1.Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) to boost performance.

2.Visualization of ROC curves, confusion matrices, and feature importances.

3.Deployment wrapping: package as a CLI, Python package, or REST API.

4.Comprehensive documentation: finalize this README and add a detailed project report.

© 2025 CardioNet-Validator Project. Logged by Sayad Ibne Azad.