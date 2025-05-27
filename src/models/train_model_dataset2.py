from src.models.model_utils import *
from sklearn.model_selection import train_test_split

df = load_data("data/processed/clean_dataset2.csv")
print("Dataset loaded:", df.shape)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = train_models(X_train, y_train)
results = evaluate_models(models, X_test, y_test)

for name, result in results.items():
    print(f"\nModel: {name}")
    print("Classification Report:")
    print(result["report"])
    print("Confusion Matrix:")
    print(result["confusion_matrix"])

save_models(models, "outputs/models_dataset2/")
