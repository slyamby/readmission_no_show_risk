"""
Feature engineering for the No-Show Prediction model.
Takes the cleaned dataframe from src.load_data and produces the model-ready
feature matrix, target, and the sensitive columns held aside for fairness auditing.
"""
import pandas as pd

FAIRNESS_AUDIT_COLS = ["gender", "scholarship"]

FEATURE_COLS = [
    "age", "neighbourhood_grouped", "hypertension", "diabetes",
    "alcoholism", "handicap_flag", "sms_received", "lead_days",
    "appointment_weekday",
]

RARE_NEIGHBOURHOOD_THRESHOLD = 100


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binarize handicap - levels 2-4 are too sparse to trust individually
    df["handicap_flag"] = (df["handicap"] > 0).astype(int)

    # Bucket rare neighbourhoods into "Other"
    neigh_counts = df["neighbourhood"].value_counts()
    rare_neighbourhoods = neigh_counts[neigh_counts < RARE_NEIGHBOURHOOD_THRESHOLD].index
    df["neighbourhood_grouped"] = df["neighbourhood"].where(
        ~df["neighbourhood"].isin(rare_neighbourhoods), "Other"
    )

    # Day of week the appointment falls on
    df["appointment_weekday"] = df["AppointmentDay"].dt.day_name()

    return df


def get_X_y_audit(df: pd.DataFrame):
    """Returns (X, y, audit_df) - audit_df holds gender/scholarship, NOT used in training."""
    df = build_features(df)
    X = df[FEATURE_COLS]
    y = df["no_show_flag"]
    audit_df = df[FAIRNESS_AUDIT_COLS]
    return X, y, audit_df