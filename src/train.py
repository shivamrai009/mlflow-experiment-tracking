import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import os

# 1. Load dataset
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Set MLflow experiment
mlflow.set_experiment("Iris-MLflow-From-Scratch")

# 3. Start MLflow run
with mlflow.start_run():

    # Hyperparameters
    C = 1.0
    max_iter = 200

    # 4. Train model
    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(X_train, y_train)

    # 5. Predictions
    preds = model.predict(X_test)

    # 6. Metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    # 7. Log parameters
    mlflow.log_param("C", C)
    mlflow.log_param("max_iter", max_iter)

    # 8. Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # 9. Save plot
    os.makedirs("artifacts", exist_ok=True)
    plt.figure()
    plt.bar(["Accuracy", "F1"], [acc, f1])
    plt.title("Model Performance")
    plt.savefig("artifacts/metrics.png")

    # 10. Log artifact
    mlflow.log_artifact("artifacts/metrics.png")

    # 11. Log model
    mlflow.sklearn.log_model(model, "model")

print("✅ Training completed successfully")
