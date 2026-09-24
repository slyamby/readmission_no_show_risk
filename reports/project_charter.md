## Project Charter — Appointment No-Show Prediction
1. Why is this project important?

No-shows are a quiet but expensive problem for outpatient clinics — every missed appointment is a wasted slot that could have gone to another patient, plus sunk staff/room time that was already allocated. Industry estimates commonly cite no-show rates of 15-30% in outpatient settings, and clinics often respond either generically (blanket SMS reminders to everyone) or not at all. A model that scores which patients are likely to miss their appointment lets a clinic intervene selectively — a phone call instead of a text, overbooking high-risk slots, or offering an earlier alternative — which is more cost-effective than treating every patient the same way.

2. Whose job will be affected?

Most directly, this touches front-desk/scheduling staff and clinic operations managers — they'd be the ones acting on the risk score (deciding who gets a reminder call vs. a text, or which slots to double-book). It could also touch nurses/clinicians indirectly, since a better-utilized schedule means less idle time between patients. The people most affected by the outcome, though, are the patients themselves — especially since some of the available features (a welfare/scholarship flag, chronic conditions like hypertension or diabetes) double as proxies for socioeconomic status and health burden. If a high no-show score ever got used punitively (e.g., deprioritizing or over-scrutinizing certain patients) rather than supportively (extra reminder outreach, transportation assistance), that would be a real harm — so this is a case where we should watch for the model leaning on sensitive features and think about how the score should and shouldn't be used downstream.

3. Do we have the right data?

Mostly yes, with real caveats worth naming upfront. The dataset (~110K appointments from Brazil's public health system, 2016) gives us the core ingredients: patient demographics (age, gender, neighborhood), chronic condition flags (hypertension, diabetes, alcoholism, handicap), a welfare/scholarship flag, whether an SMS reminder was sent, and the scheduling lead time (days between booking and the appointment) — one of the strongest known predictors of no-shows in this literature.

Caveats: it's a single city/region from 2016, so a model trained on it wouldn't transfer directly to a different clinic or country without retraining. It's also known to have data quality issues (a few negative age values, some scheduling timestamps that don't make logical sense) that need explicit handling, not silent dropping. It's missing features that would likely help in a real deployment — appointment type/specialty, distance/transport time to the clinic, whether it's a first visit vs. returning patient.

4. What determines the project is done?

Since this is a portfolio project rather than a live deployment, "done" is about a complete, defensible body of work rather than hitting a specific accuracy target:

(a) Data cleaned with known quality issues explicitly handled and documented, not silently dropped.
(b) An EDA write-up identifying which factors actually separate no-shows from shows.
(c) A model that meaningfully beats a naive baseline (e.g., "always predict shows up," or "flag everyone with a long lead time") on a metric reflecting the real cost trade-off — likely recall on no-shows, since missing a likely no-show is costlier for a clinic than a false alarm.
(d) A fairness check on sensitive attributes (age, gender, welfare status) given who's affected (Q2).
(e) An interactive Streamlit dashboard a non-technical scheduler could actually use.
(f) The repo in a production-style structure (data/src/utils/models/notebooks/reports/logs/config), plus a case-study write-up.
5. What if the outcome isn't what we expected?
If the model barely beats the naive baseline, we report that honestly and pivot the value-add toward whichever weak signals do hold up (e.g., lead time), rather than overselling accuracy.
If the fairness audit turns up a real bias risk (e.g., age or the welfare flag driving the score in a way that could look like discriminatory triage), we handle it the same way as the Credit Risk project: consider excluding it from training and auditing post-hoc, and document the trade-off rather than hiding it.
If data quality issues (negative ages, inconsistent timestamps) affect a meaningful chunk of rows rather than a handful, we document exactly how much was affected and how we handled it, instead of quietly dropping rows and moving on.

The commitment here isn't a technical contingency so much as reporting the real result — including a null or uncomfortable one — rather than reverse-engineering the narrative.