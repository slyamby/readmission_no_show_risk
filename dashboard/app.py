"""
Streamlit dashboard for the Appointment No-Show Prediction model.
Loads the pre-trained pipeline and pre-scored test set — no retraining happens here.
"""

import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.metrics import roc_auc_score


MODEL_PATH = "models/no_show_pipeline.joblib"
SCORED_TEST_PATH = "data/processed/scored_test_set.csv"

st.set_page_config(page_title="No-show Risk Dashboard", layout="wide")


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_scored_data():
    return pd.read_csv(SCORED_TEST_PATH)


pipeline = load_pipeline()
scored = load_scored_data()

st.title("Appointment No-Show Risk Dashboard")
st.caption("Model: XGBoost | Test set: {:,} appointments".format(len(scored)))

# --- KPI row ---
actual_rate = scored["actual_no_show"].mean()
avg_predicted_risk = scored["predicted_risk"].mean()
test_auc = roc_auc_score(scored["actual_no_show"], scored["predicted_risk"])

high_risk_count = (scored["predicted_risk"] >= scored["predicted_risk"].quantile(0.67)).sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Test Set Size", f"{len(scored):,}")
col2.metric("Actual No-Show Rate", f"{actual_rate:.1%}")
col3.metric("Model AUC", f"{test_auc:.3f}")
col4.metric("High-Risk Patients", f"{high_risk_count:,}")

# --- Risk tier breakdown ---
st.subheader("Risk Tiers vs. Actual Outcomes")
tier_labels = ["Low", "Medium", "High"]
scored["risk_tier"] = pd.qcut(scored["predicted_risk"],q=3, labels=tier_labels)

tier_summary = (
    scored.groupby("risk_tier", observed=True)
    .agg(actual_rate=("actual_no_show", "mean"), n=("actual_no_show", "size"))
    .reindex(tier_labels)
    .reset_index()
)

tier_colors = {"Low": "#BFDBFE", "Medium": "#60A5FA", "High": "#1D4ED8"}

fig_tiers = px.bar(
    tier_summary,
    x="risk_tier",
    y="actual_rate",
    color="risk_tier",
    color_discrete_map=tier_colors,
    text=tier_summary.apply(lambda r: f"{r['actual_rate']:.1%}  (n={r['n']:,})", axis=1),
)

fig_tiers.update_traces(textposition="outside", textfont_color="#1F2937")
fig_tiers.add_hline(
    y=actual_rate,
    line_dash="dash",
    line_color="#6B7280",
    annotation_text=f"Overall rate: {actual_rate:.1%}",
    annotation_position="top left",
)
fig_tiers.update_layout(
    showlegend=False,
    yaxis_tickformat=".0%",
    xaxis_title=None,
    yaxis_title="Actual No-Show Rate",
    plot_bgcolor="white",
    paper_bgcolor="white",
)
st.plotly_chart(fig_tiers, use_container_width=True)

# --- Score distribution ---
st.subheader("Predicted Risk Distribution by Actual Outcome")

dist_data = scored.copy()
dist_data["Outcome"] = dist_data["actual_no_show"].map({0: "Showed", 1: "No-Show"})

outcome_colors = {"Showed": "#94A3B8", "No-Show": "#F59E0B"}

fig_dist = px.histogram(
    dist_data,
    x="predicted_risk",
    color="Outcome",
    color_discrete_map=outcome_colors,
    barmode="overlay",
    opacity=0.65,
    nbins=40,
    category_orders={"Outcome": ["Showed", "No-Show"]},
)
fig_dist.update_layout(
    xaxis_title="Predicted Risk Score",
    yaxis_title="Number of Patients",
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend_title_text="",
)
st.plotly_chart(fig_dist, use_container_width=True)

# --- Feature importance ---
st.subheader("What Drives the Model's Predictions")

feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = pipeline.named_steps["classifier"].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances,
}).sort_values("importance", ascending=False).head(15)

# Clean up sklearn's prefixed names (num__, bin__, cat__) for readability
importance_df["feature"] = (
    importance_df["feature"]
    .str.replace(r"^(num|bin|cat)__", "", regex=True)
    .str.replace("_", " ")
    .str.title()
)

fig_importance = px.bar(
    importance_df.sort_values("importance"),
    x="importance",
    y="feature",
    orientation="h",
)
fig_importance.update_traces(marker_color="#3B82F6")
fig_importance.update_layout(
    xaxis_title="Relative Importance",
    yaxis_title=None,
    plot_bgcolor="white",
    paper_bgcolor="white",
    showlegend=False,
)
st.plotly_chart(fig_importance, use_container_width=True)

st.caption(
    "Built-in XGBoost feature importance (overall usage across the model). "
    "For a per-prediction, direction-aware explanation, see the SHAP analysis in the model results report."
)

# --- Feature importance (aggregated back to original features) ---
st.subheader("What Drives the Model's Predictions")

preprocessor = pipeline.named_steps["preprocessor"]
categorical_cols = next(t[2] for t in preprocessor.transformers_ if t[0] == "cat")

feature_names = preprocessor.get_feature_names_out()
importances = pipeline.named_steps["classifier"].feature_importances_

raw_df = pd.DataFrame({"feature": feature_names, "importance": importances})


def to_original_feature(name):
    name = name.split("__", 1)[-1]  # drop sklearn's num__/bin__/cat__ prefix
    for col in categorical_cols:  # one-hot encoded columns collapse back to their source
        if name.startswith(col):
            return col
    return name  # numeric/binary features are already one column each


raw_df["original_feature"] = raw_df["feature"].apply(to_original_feature)

importance_df = (
    raw_df.groupby("original_feature")["importance"]
    .mean()
    .reset_index()
    .sort_values("importance", ascending=False)
    .head(10)
)
importance_df["original_feature"] = (
    importance_df["original_feature"].str.replace("_", " ").str.title()
)

fig_importance = px.bar(
    importance_df.sort_values("importance"),
    x="importance",
    y="original_feature",
    orientation="h",
)
fig_importance.update_traces(marker_color="#3B82F6")
fig_importance.update_layout(
    xaxis_title="Relative Importance",
    yaxis_title=None,
    plot_bgcolor="white",
    paper_bgcolor="white",
    showlegend=False,
)
st.plotly_chart(fig_importance, use_container_width=True)

st.caption(
    "Built-in XGBoost feature importance, averaged across one-hot encoded columns "
    "within each original feature (e.g. neighbourhood) so high-cardinality features "
    "aren't inflated just for having more categories. This is a coarse, global view — "
    "for the more reliable per-prediction, direction-aware explanation, see the SHAP "
    "analysis in the model results report."
)

# --- Fairness audit ---
st.subheader("Fairness Audit: Gender & Scholarship")
st.caption(
    "Gender and scholarship (welfare enrollment) were excluded from training, then checked "
    "against the model's predictions afterward. A widening gap from left to right would mean "
    "the model amplifies a disparity beyond what's actually in the data."
)


def fairness_summary(df, group_col, group_labels=None):
    summary = (
        df.groupby(group_col)
        .agg(
            actual_rate=("actual_no_show", "mean"),
            avg_predicted_risk=("predicted_risk", "mean"),
            classified_rate=("predicted_no_show", "mean"),
        )
        .reset_index()
    )
    if group_labels:
        summary[group_col] = summary[group_col].map(group_labels)
    return summary


def render_fairness_chart(summary_df, group_col, color_map):
    metric_labels = {
        "actual_rate": "Actual No-Show Rate",
        "avg_predicted_risk": "Avg Predicted Risk",
        "classified_rate": "Classified No-Show Rate",
    }
    long_df = summary_df.melt(
        id_vars=group_col,
        value_vars=list(metric_labels.keys()),
        var_name="metric",
        value_name="rate",
    )
    long_df["metric"] = long_df["metric"].map(metric_labels)

    fig = px.bar(
        long_df,
        x="metric",
        y="rate",
        color=group_col,
        barmode="group",
        color_discrete_map=color_map,
        category_orders={"metric": list(metric_labels.values())},
        text=long_df["rate"].apply(lambda v: f"{v:.1%}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis_tickformat=".0%",
        xaxis_title=None,
        yaxis_title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True)


col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**By Gender**")
    gender_summary = fairness_summary(scored, "gender", {"M": "Male", "F": "Female"})
    render_fairness_chart(gender_summary, "gender", {"Male": "#22C55E", "Female": "#A855F7"})

with col_b:
    st.markdown("**By Scholarship (Welfare Enrollment)**")
    scholarship_summary = fairness_summary(
        scored, "scholarship", {0: "Not Enrolled", 1: "Enrolled"}
    )
    render_fairness_chart(
        scholarship_summary, "scholarship", {"Not Enrolled": "#22C55E", "Enrolled": "#A855F7"}
    )

# --- Interactive scorer ---
st.subheader("Score a Hypothetical Patient")

with st.form("scorer_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=115, value=35)
        lead_days = st.number_input("Lead Time (days)", min_value=0, max_value=200, value=10)
        neighbourhood = st.selectbox("Neighbourhood", sorted(scored["neighbourhood_grouped"].unique()))
        weekday = st.selectbox("Appointment Weekday", sorted(scored["appointment_weekday"].unique()))
    with col2:
        hypertension = st.checkbox("Hypertension")
        diabetes = st.checkbox("Diabetes")
        alcoholism = st.checkbox("Alcoholism")
        handicap_flag = st.checkbox("Handicap")
        sms_received = st.checkbox("SMS Reminder Received")

    submitted = st.form_submit_button("Score Patient")

if submitted:
    input_df = pd.DataFrame([{
        "age": age,
        "lead_days": lead_days,
        "hypertension": int(hypertension),
        "diabetes": int(diabetes),
        "alcoholism": int(alcoholism),
        "handicap_flag": int(handicap_flag),
        "sms_received": int(sms_received),
        "neighbourhood_grouped": neighbourhood,
        "appointment_weekday": weekday,
    }])

    risk = pipeline.predict_proba(input_df)[0, 1]
    prediction = pipeline.predict(input_df)[0]

    risk = pipeline.predict_proba(input_df)[0, 1]
    prediction = pipeline.predict(input_df)[0]

    st.metric("Predicted No-Show Risk", f"{risk:.1%}")
    st.caption("Remember: this score is a ranking tool, not a calibrated probability — treat it as relative risk, not a literal chance of no-show.")

    if prediction == 1:
        st.warning("Model flags this patient as likely to no-show — consider a phone call reminder or an earlier-slot offer.")
    else:
        st.success("Model does not flag this patient as high-risk.")