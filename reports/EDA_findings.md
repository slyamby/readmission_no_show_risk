# EDA Findings — Appointment No-Show Prediction

**Dataset:** Medical Appointment No Shows ([Kaggle](https://www.kaggle.com/datasets/joniarroba/noshowappointments)) — 110,527 outpatient appointments from Brazil's public health system (Vitória, Espírito Santo), May 2016. 14 raw columns; 110,521 rows and 16 columns after cleaning.

## Key Findings

No-shows happen in about 1 out of every 5 scheduled appointments (20.2%), and the single strongest signal we found is how far in advance the appointment was booked: patients who missed their appointment had waited an average of 15.8 days between booking and the visit, compared to just 8.75 days for patients who showed up — nearly double. No-show patients also skew slightly younger (34.3 years old on average, versus 37.8 for those who showed) and are modestly more likely to be enrolled in the welfare/scholarship program (23.7% no-show rate versus 19.8%), which is worth watching carefully later since it doubles as a socioeconomic signal, not just a behavioral one. Chronic conditions point the opposite direction from what some might expect — patients with hypertension or diabetes actually show up *more* reliably (17-18% no-show vs. 20-21% for those without), plausibly because managing an ongoing condition means more practice keeping appointments. One result looks paradoxical at first glance — patients who received an SMS reminder had a *higher* no-show rate (27.6% vs 16.7%) — but this is a confound, not reminders backfiring: SMS reminders can only be sent when there's enough lead time to send them, and we already know longer lead time itself predicts more no-shows. Geography (neighbourhood) shows real variation among well-sampled areas (up to ~29% no-show in the highest neighbourhoods, versus the 20.2% baseline), but several small neighbourhoods show misleadingly extreme rates (including 100%) driven by having only 1-2 appointments on record, not a real effect.

## Load & Profile

- Raw shape: 110,527 rows × 14 columns. No missing values in any column. No duplicate rows; no duplicate `AppointmentID` values (confirms it's a valid unique key).
- After cleaning: 110,521 rows × 16 columns (6 rows dropped — see Data Quality Notes; 2 derived columns added: `lead_days`, `no_show_flag`).
- Target: `no_show` (Yes/No), also encoded numerically as `no_show_flag`. No-show rate: **20.19%** — in line with published outpatient no-show rates generally cited in the 15-30% range.

## Univariate

- **Age**: right-skewed, 0-115. A distinct spike near age 0 corresponds to pediatric appointments (likely vaccinations/well-baby visits) rather than a data error. Max of 115 is biologically plausible and was not treated as an outlier.
- **Lead time (`lead_days`)**: heavily right-skewed — a slight majority of appointments are booked same-day (`lead_days = 0`), with a long tail out to 179 days.
- **Handicap**: documented as boolean in the source but actually stores a count (0-4). Overwhelmingly concentrated at 0 (108,286 of 110,521 rows); values 3 and 4 have only 13 and 3 records respectively.
- **Neighbourhood**: 81 unique values, ranging from 1 appointment (Parque Industrial) up to the low thousands for the largest neighbourhoods — a classic long-tail categorical.

## Bivariate — Relationship to No-Show

| Feature | No-show group | Show group |
|---|---|---|
| Mean lead time (days) | 15.84 | 8.75 |
| Mean age (years) | 34.3 | 37.8 |

No-show rate by binary flag:

| Feature | Rate when 1/positive | Rate when 0/negative |
|---|---|---|
| Scholarship (welfare) | 23.7% | 19.8% |
| Hypertension | 17.3% | 20.9% |
| Diabetes | 18.0% | 20.4% |
| Alcoholism | 20.1% | 20.2% |
| SMS received | 27.6% | 16.7% |
| Gender (F vs M) | 20.3% (F) | 20.0% (M) |

`handicap` shows an apparent upward trend in no-show rate (17.8% → 20.2% → 23.1% → 33.3% across levels 1-4), but levels 3 and 4 are backed by only 13 and 3 appointments respectively — too small to trust as a real effect.

`neighbourhood`: the largest, best-sampled neighbourhoods (Santos Dumont, Santa Cecília, Santa Clara, Itararé) all show no-show rates of 26-29%, meaningfully above the 20.2% baseline — a plausible real geographic effect. Several of the smallest neighbourhoods show 0% or 100% no-show rates, driven entirely by sample sizes of 1-2 appointments.

## Data Quality Notes

- **Scheduling date logic (resolved)**: comparing `AppointmentDay` and `ScheduledDay` as full timestamps flagged 38,568 rows (35% of the dataset) where the appointment appeared to be "before" the scheduling date. This was a false signal: `AppointmentDay` is stored at midnight (00:00:00) while `ScheduledDay` retains a full timestamp, so same-day bookings look like violations under a naive comparison. Re-comparing at the date level (not timestamp level) found only **5 genuine violations**, which were dropped along with the false 38,563 apparent violations correctly reclassified as valid.
- **Negative age (resolved)**: 1 row had `Age = -1`, an impossible value. Dropped.
- **Net rows dropped**: 6 of 110,527 (5 genuine date-logic violations + 1 negative age) — under 0.01% of the data, handled by dropping rather than imputing, since the volume is negligible and the values are unrecoverable errors rather than plausible-but-missing data.
- **Column naming (resolved)**: source columns had inconsistent/misspelled names (`Hipertension`, `Handcap`, `No-show` with a hyphen, mixed casing). Standardized to `hypertension`, `handicap`, `no_show`, and snake_case throughout for consistency with the rest of the codebase.
- **`Handicap` is a count, not boolean (noted for feature engineering)**: despite the dataset documentation implying a yes/no flag, values range 0-4. Given how sparse levels 2-4 are, we plan to collapse this to a binary (any handicap vs. none) rather than model it as an ordinal/continuous count.
- **High-cardinality `neighbourhood` (noted for feature engineering)**: 81 categories with a long tail down to single-digit appointment counts. Rare categories produce misleading no-show rates purely from small samples. Plan: group any neighbourhood below a minimum appointment threshold (e.g., <100) into an "Other" bucket before encoding, to avoid the model learning from noise.
- **`SMS_received` confound (noted for modeling)**: raw no-show rate is higher when an SMS was received, but this is very likely explained by `lead_days` (SMS reminders require lead time to send) rather than reminders causing no-shows. Any interpretation of the SMS feature's effect should control for lead time rather than reading the raw rate at face value.
