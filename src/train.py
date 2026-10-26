"""
Train the final real-estate price prediction model.

Run from the project root:
    python src/train.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_data.csv"
CONFIG_PATH = PROJECT_ROOT / "models" / "feature_config.json"
MODEL_PATH = PROJECT_ROOT / "models" / "selected_final_model.joblib"


# ============================================================
# 2. LOAD FEATURE CONFIGURATION
# ============================================================

def load_feature_config():
    """Load numerical and categorical feature names."""

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]

    return numeric_features, categorical_features


# ============================================================
# 3. BUILD PREPROCESSOR
# ============================================================

def build_preprocessor(numeric_features, categorical_features):
    """Build preprocessing pipelines for numerical and categorical data."""

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


# ============================================================
# 4. BUILD FINAL MODEL
# ============================================================

def build_model(numeric_features, categorical_features):
    """Build the selected Random Forest model."""

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    regression_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    max_features=0.7,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Train on log-transformed prices and return predictions
    # in the original price scale.
    final_model = TransformedTargetRegressor(
        regressor=regression_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    return final_model


# ============================================================
# 5. LOAD DATA AND PREPARE TRAINING SETS
# ============================================================

def load_data():
    """Load processed data and separate training and test sets."""

    df = pd.read_csv(DATA_PATH)

    numeric_features, categorical_features = load_feature_config()

    feature_columns = numeric_features + categorical_features

    required_columns = (
        feature_columns
        + ["price_lakhs", "split"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing from the dataset: "
            f"{missing_columns}"
        )

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Training or test data is empty. Check the split column."
        )

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()

    y_train = train_df["price_lakhs"].copy()
    y_test = test_df["price_lakhs"].copy()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        numeric_features,
        categorical_features,
    )


# ============================================================
# 6. TRAIN, EVALUATE, AND SAVE
# ============================================================

def main():
    """Run the complete model training workflow."""

    print("=" * 60)
    print("REAL-ESTATE PRICE PREDICTION: MODEL TRAINING")
    print("=" * 60)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        numeric_features,
        categorical_features,
    ) = load_data()

    print(f"\nTraining rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}")
    print(f"Numerical features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Total input features: {len(numeric_features) + len(categorical_features)}")

    # Build the final model.
    model = build_model(
        numeric_features,
        categorical_features,
    )

    # Fit using training data only.
    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    print("Training completed successfully!")

    # Predict prices in lakhs.
    predictions = model.predict(X_test)

    # Evaluate on the untouched test split.
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\nTEST SET PERFORMANCE")
    print("-" * 40)
    print(f"R²:         {r2:.4f}")
    print(f"MAE:        ₹{mae:.4f} lakhs")
    print(f"RMSE:       ₹{rmse:.4f} lakhs")

    # Save the complete preprocessing + model pipeline.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print("\nModel saved successfully!")
    print(f"Location: {MODEL_PATH}")

    # Verify that the saved model can be loaded.
    loaded_model = joblib.load(MODEL_PATH)

    sample_predictions = loaded_model.predict(X_test.head(5))

    print("\nMODEL RELOAD VERIFICATION")
    print("-" * 40)
    print("Successfully loaded the saved model.")
    print("Sample predicted prices (lakhs):")
    print(np.round(sample_predictions, 2))

    print("\nTraining workflow completed successfully!")


# ============================================================
# 7. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()