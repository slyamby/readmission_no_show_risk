"""
Load and clean the Medical Appointment No-Show dataset.
Source: https://www.kaggle.com/datasets/joniarroba/noshowappointments
"""
import pandas as pd

RAW_PATH = "data/raw/KaggleV2-May-2016.csv"
PROCESSED_PATH = "data/processed/no_show_clean.csv"

RENAME_MAP = {
    "PatientId": "patient_id",
    "AppointmentID": "appointment_id",
    "Gender": "gender",
    "Age": "age",
    "Neighbourhood": "neighbourhood",
    "Scholarship": "scholarship",
    "Hipertension": "hypertension",
    "Diabetes": "diabetes",
    "Alcoholism": "alcoholism",
    "Handcap": "handicap",
    "SMS_received": "sms_received",
    "No-show": "no_show",
}


def load_and_clean(raw_path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])

    # Date-only comparison (AppointmentDay is stored at midnight while
    # ScheduledDay keeps time-of-day, so comparing full timestamps produces
    # ~38K false "violations" that are just same-day bookings)
    scheduled_date = df["ScheduledDay"].dt.normalize()
    appointment_date = df["AppointmentDay"].dt.normalize()

    # Drop genuine data errors: true date-level scheduling violations (5 rows)
    # and the single negative-age row. ~0.01% of the data - safe to drop.
    valid = (appointment_date >= scheduled_date) & (df["Age"] >= 0)
    df = df[valid].copy()
    scheduled_date, appointment_date = scheduled_date[valid], appointment_date[valid]

    df = df.rename(columns=RENAME_MAP)
    df["patient_id"] = df["patient_id"].astype("int64")
    df["lead_days"] = (appointment_date - scheduled_date).dt.days
    df["no_show_flag"] = (df["no_show"] == "Yes").astype(int)

    return df


if __name__ == "__main__":
    df = load_and_clean()
    df.to_csv(PROCESSED_PATH, index=False)
    print("Shape:", df.shape)
    print("\nDtypes:\n", df.dtypes)
    print("\nNo-show rate:", df["no_show_flag"].mean())