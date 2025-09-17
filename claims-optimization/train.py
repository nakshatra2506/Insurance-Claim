import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from pathlib import Path
import warnings

DATA_PATH = Path("data/claims.csv")
MODEL_PATH = Path("models/claim_approval.joblib")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Missing data/claims.csv. Use your CSV or run generate_sample_data.py")

    df = pd.read_csv(DATA_PATH)

    if "status" not in df.columns:
        raise ValueError("Training requires a 'status' column with values Approved/Denied.")
    y = (df["status"].astype(str).str.lower() == "approved").astype(int)
    X = df.drop(columns=["status"])

    # Edit these if your headers differ:
    num_cols  = ["claim_amount","patient_age","prior_denials_count","days_since_service"]
    bool_cols = ["in_network","preauth_obtained"]
    cat_cols  = ["patient_gender","insurance_provider","diagnosis_code","procedure_code"]

    for col in num_cols + bool_cols + cat_cols:
        if col not in X.columns:
            warnings.warn(f"Column '{col}' not found in data; update feature lists.")

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), [c for c in num_cols if c in X.columns]),
        ("bool", "passthrough", [c for c in bool_cols if c in X.columns]),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=20))
        ]), [c for c in cat_cols if c in X.columns]),
    ], remainder="drop")

    base = LogisticRegression(max_iter=300, class_weight="balanced")
    clf = Pipeline([
        ("pre", pre),
        ("cal_lr", CalibratedClassifierCV(base, method="isotonic", cv=3))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:,1]
    roc = roc_auc_score(y_test, proba)
    pr  = average_precision_score(y_test, proba)
    print(f"ROC-AUC: {roc:.3f} | PR-AUC (Approved): {pr:.3f}")

    y_pred = (proba >= 0.5).astype(int)
    print(classification_report(y_test, y_pred, target_names=["Denied","Approved"]))

    meta = {"features": {"num": num_cols, "bool": bool_cols, "cat": cat_cols},
            "threshold_default": 0.5, "target": "status", "positive_class": "Approved"}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "meta": meta}, MODEL_PATH)
    print(f"Saved {MODEL_PATH.resolve()}")

if __name__ == "__main__":
    main()
