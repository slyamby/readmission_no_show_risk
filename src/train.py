"""
Train the final No-Show Prediction model and persist the artifacts
the dashboard needs: the fitted pipeline and a scored test set.
"""
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from load_data import load_and_clean
from features import get_X_y_audit

RAW_PATH = "data/raw/KaggleV2-May-2016.csv"
MODEL_PATH = "models/no_show_pipeline.joblib"
SCORED_TEST_PATH = "data/processed/scored_test_set.csv"
LOG_PATH = "logs/train.log"

NUMERIC_FEATURES = ["age", "lead_days"]
BINARY_FEATURES = ["hypertension", "diabetes", "alcoholism", "handicap_flag", "sms_received"]
CATEGORICAL_FEATURES = ["neighbourhood_grouped", "appointment_weekday"]

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(random_state=42, eval_metric="logloss")),
        # scale_pos_weight is set at fit time, after we know the train split
    ])


def main():
    logger.info("Loading and cleaning raw data from %s", RAW_PATH)
    df = load_and_clean(RAW_PATH)
    logger.info("Loaded %d rows after cleaning", len(df))

    X, y, audit_df = get_X_y_audit(df)
    logger.info("Built feature matrix: %d rows, %d columns", X.shape[0], X.shape[1])

    logger.info("Splitting into train/test (80/20, stratified on target)")
    X_train, X_test, y_train, y_test, audit_train, audit_test = train_test_split(
        X, y, audit_df, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train shape: %s | Test shape: %s", X_train.shape, X_test.shape)

    pipeline = build_pipeline()
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    pipeline.set_params(classifier__scale_pos_weight=scale_pos_weight)

    logger.info("Fitting XGBoost pipeline (scale_pos_weight=%.3f)...", scale_pos_weight)
    pipeline.fit(X_train, y_train)
    logger.info("Model fit complete.")

    joblib.dump(pipeline, MODEL_PATH)
    logger.info("Model saved to: %s", MODEL_PATH)

    scored = X_test.copy()
    scored["actual_no_show"] = y_test
    scored["predicted_risk"] = pipeline.predict_proba(X_test)[:, 1]
    scored["predicted_no_show"] = pipeline.predict(X_test)
    scored["gender"] = audit_test["gender"]
    scored["scholarship"] = audit_test["scholarship"]
    scored.to_csv(SCORED_TEST_PATH, index=False)
    logger.info("Scored test set saved to: %s", SCORED_TEST_PATH)

    test_auc = roc_auc_score(y_test, scored["predicted_risk"])
    logger.info(
        "Test AUC: %.3f (%s trained on %d rows)",
        test_auc,
        pipeline.named_steps["classifier"].__class__.__name__,
        X_train.shape[0],
    )


if __name__ == "__main__":
    main()