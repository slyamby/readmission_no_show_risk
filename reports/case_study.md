# Appointment No-Show Prediction — Case Study Content

Three ready-to-use variants, all pulled from the actual project results (Medical Appointment No-Shows dataset, ~110,500 records, XGBoost, AUC 0.734).

---

## (a) GitHub README — "Results" section

## Results

I built a model to predict which scheduled medical appointments are likely to end in a no-show, using ~110,500 outpatient records from Brazil's public health system (2016). About 1 in 5 appointments (20.2%) are no-shows in this data, and clinics that can't tell which ones in advance end up treating every patient the same way — generic reminders, no prioritization.

| Model | AUC | No-Show Recall | No-Show Precision | Accuracy |
|---|---|---|---|---|
| **XGBoost (selected)** | **0.734** | **77%** | 32% | 62% |
| Logistic Regression | 0.665 | 58% | 31% | 65% |

XGBoost outperformed Logistic Regression here — the reverse of what I found on an earlier, much smaller project (Credit Risk Scoring), where Logistic Regression won on ~800 rows. With ~88,000 training rows, XGBoost's extra capacity pays off instead of overfitting — a concrete illustration that model choice depends on data volume, not a fixed hierarchy of algorithms.

The score's real value shows up when patients are grouped by predicted risk: the highest-risk third of patients actually no-show at **35%**, versus just **4.2%** for the lowest-risk third — an 8x spread a scheduler could act on directly (a phone call instead of a text, an earlier-slot offer, or targeted transportation assistance).

**Approach:**
- Cleaned raw scheduling data, catching a timestamp-logic bug that initially made 35% of rows look like invalid bookings before tracing it to a data-formatting artifact (same-day appointments), not a real quality issue — only 5 rows were genuine errors.
- Engineered features from the raw dataset (booking lead time, appointment weekday, a grouped/de-noised neighbourhood category) and addressed a 20/80 class imbalance with class-weighting.
- Used **SHAP** to explain predictions in plain language: lead time and age were the two strongest drivers, and — counterintuitively — patients with chronic conditions like hypertension showed up *more* reliably, not less.
- Ran a **fairness audit**: excluded gender and welfare-enrollment status from training, then checked predictions against them after the fact. Traced a real disparity by welfare status back to an age-driven proxy effect; found — and reported, rather than hid — an unresolved gender-based disparity in predicted risk that the available features couldn't fully explain.
- Shipped an interactive **Streamlit dashboard** with risk-tier breakdowns, a live fairness-audit view, and a scorer for hypothetical patients, so a non-technical scheduler could use the model directly.

**Caveat (stated up front, not hidden):** predicted risk scores are a valid ranking but not calibrated probabilities — class-weighting used to handle the imbalance inflates them above the true ~20% base rate. The model was built and evaluated on a single city and year, and an unexplained gender-based prediction gap is flagged as an open issue rather than papered over.

---

## (b) LinkedIn / portfolio post (under 200 words)

Just wrapped up a project predicting medical appointment no-shows, and the part I want to highlight isn't the accuracy number.

Using ~110,000 appointment records from a public health system, I built an XGBoost model (AUC 0.734) that sorts patients by no-show risk. The useful part: patients in the top third by predicted risk actually miss their appointment 35% of the time, versus just 4.2% for the bottom third — a gap a scheduler could act on with a phone call or an earlier slot offer, instead of treating every patient the same way.

The part I'm proud of is the fairness audit. I deliberately excluded gender and welfare-enrollment status from training, then checked the model's predictions against those groups afterward. One disparity traced cleanly back to an age-related proxy effect. The other — a gender-based gap with no real-world outcome difference behind it — I couldn't fully explain, so I reported it as an open issue instead of quietly smoothing it over.

I also shipped an interactive Streamlit dashboard with a live fairness view and a scorer, so a non-technical scheduler could actually use this.

Code, write-up, and dashboard link in comments.

---

## (c) Resume bullet

Built an XGBoost model (0.734 AUC, scikit-learn) to predict medical appointment no-shows from 110K+ records, achieving 77% recall and separating patients into risk tiers with an 8x spread in real no-show rates; ran a post-hoc fairness audit that surfaced and reported an unresolved demographic disparity, and shipped an interactive Streamlit dashboard for non-technical schedulers.
