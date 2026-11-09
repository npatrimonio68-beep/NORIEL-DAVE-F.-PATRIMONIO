"""
Student Performance Prediction System
Downloads the UCI Student Performance dataset, trains a Random Forest model,
evaluates it, and saves student_data.csv and student_model.pkl.
"""
import pandas as pd
import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)

RANDOM_STATE = 42

def main():
    print("Downloading UCI Student Performance dataset...")
    student = fetch_ucirepo(id=320)

    # Combine student attributes and Mathematics grades.
    df = pd.concat(
        [student.data.features, student.data.targets],
        axis=1
    )
    df = df.drop_duplicates().copy()

    if "G3" not in df.columns:
        raise ValueError("The downloaded dataset does not contain the G3 target column.")

    df.to_csv("student_data.csv", index=False)
    print(f"Dataset saved. Shape: {df.shape}")
    print("\nFirst five rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())

    # 1 = AT RISK when G3 < 10; 0 = NOT AT RISK otherwise.
    y = (df["G3"] < 10).astype(int)

    # G3 is the target only. Never include it among model inputs.
    X = df.drop(columns=["G3"])

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ))
    ])

    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("\nMODEL EVALUATION")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions,
        labels=[0, 1],
        target_names=["Not at risk", "At risk"],
        zero_division=0
    ))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    joblib.dump(model, "student_model.pkl")
    print("\nSaved model as student_model.pkl")
    print("Training completed successfully.")

if __name__ == "__main__":
    main()
