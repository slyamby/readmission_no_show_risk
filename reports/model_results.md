# Model Results — Appointment No-Show Prediction

## Comparison Table

| Model | AUC | No-Show Recall | No-Show Precision | Accuracy |
|---|---|---|---|---|
| **XGBoost** | **0.734** | **0.77** | 0.32 | 0.62 |
| Logistic Regression | 0.665 | 0.58 | 0.31 | 0.65 |

## Best Model: XGBoost

XGBoost outperforms Logistic Regression on every metric that matters for this problem: a meaningfully higher AUC (0.734 vs. 0.665) and, more importantly, a much higher recall on the no-show class (0.77 vs. 0.58) at essentially the same precision — meaning it catches significantly more of the patients who actually go on to miss their appointment. This is a reversal of what we found on an earlier project (Credit Risk Scoring) using the same two algorithms, where Logistic Regression won on a small (~800-row) training set. Here, with roughly 88,000 training rows, XGBoost's added model capacity pays off instead of overfitting — a useful illustration that "which model wins" depends on data volume, not a fixed hierarchy of algorithms. The trade-off is that XGBoost's predictions are less directly interpretable than a linear model's coefficients; we address that with SHAP explainability below rather than defaulting back to the simpler model.

## Explainability (SHAP)

- **Lead time is one of the two strongest drivers of predicted risk**: the longer a patient waits between booking and their appointment date, the higher their predicted no-show risk. This matches the raw data directly — patients who missed appointments had waited nearly twice as long on average (15.8 days) as those who showed up (8.75 days).
- **Age is the single strongest driver**: younger patients are predicted at higher risk; older patients at lower risk.
- **Appointments booked for a Friday carry a meaningfully higher predicted risk** than other days of the week — a pattern the exploratory analysis didn't specifically test for, but one the model picked up once the feature was added.
- **Chronic conditions point the "protective" direction**: patients with hypertension or a recorded handicap are predicted at somewhat *lower* risk, not higher — plausibly because managing an ongoing condition correlates with more consistent healthcare engagement.
- **Specific neighbourhoods carry their own modest signal** beyond what age or lead time alone would predict, suggesting a real (if smaller) geographic effect — plausibly proxying for things like transportation access.
- **SMS reminders show only a small independent effect** once lead time is accounted for. The raw data showed a much larger gap (27.6% no-show rate with a reminder vs. 16.7% without), but that gap was mostly a byproduct of reminders being sent more often for appointments booked further in advance — which are already higher-risk for unrelated reasons. Once the model has lead time available directly, SMS receipt adds comparatively little on its own.

## Fairness Audit

`gender` and `scholarship` (welfare enrollment) were deliberately excluded from training, then checked against the model's predictions afterward.

- **Scholarship**: the real-world no-show gap (19.9% not enrolled vs. 23.2% enrolled — about 3.3 points) is substantially explained by age: scholarship-enrolled patients are on average 6.5 years younger, and younger age independently predicts higher no-show risk in this model. The model's average predicted-risk gap (42.6% vs. 44.6%, ~2 points) is roughly proportionate to the real disparity, but the gap at the classification level (48.6% vs. 54.1%, ~5.5 points) is wider than the real one — worth flagging if this model is ever used to make binary outreach decisions rather than continuous risk scores.
- **Gender**: there is essentially no real difference in outcomes between men and women (20.1% vs. 20.3% no-show), yet the model's predictions show a real gap anyway (43.5% vs. 41.6% average predicted risk, widening to 50.2% vs. 47.1% at the classification level). We traced this partway — women are older on average, which should *lower* predicted risk (the opposite direction), while women also have slightly longer lead times and receive more SMS reminders, both of which modestly raise predicted risk — but we could not fully attribute the gap to a single feature. This is an open caveat rather than a resolved finding.

## Caveats

- **Predicted risk scores are not calibrated probabilities.** Class-weighting (`scale_pos_weight`) was used to handle the ~20%/80% class imbalance, which inflates predicted probabilities above the true base rate. Treat the score as a ranking tool, not a literal probability of no-show.
- **Single city, single year.** The data covers one city's public health system in 2016. The model would need retraining before being trusted on a different clinic, region, or time period.
- **Known missing features.** Appointment type/specialty, distance or transport time to the clinic, and whether the visit was a first appointment vs. a returning patient are not in this dataset but would likely improve real-world performance.
- **Unresolved fairness gap on gender.** As noted above, the model produces a small but real predicted-risk disparity by gender that isn't fully explained by the available features — this should be monitored, not ignored, if the model is used operationally.
- **Intended use**: this model is meant to guide *supportive* outreach (a phone call instead of a text, transportation assistance, earlier-slot offers) toward higher-risk patients — not to deprioritize, restrict, or add friction to their care. Given the socioeconomic and demographic proxies involved, using the score punitively would risk compounding exactly the disparities this audit surfaced.
