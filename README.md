# Smartphone digital phenotyping of social anxiety disorder

Analysis code for the manuscript "Exploring Cognitive and Behavioral Characteristics of Social Anxiety Disorder through Smartphone Digital Phenotyping of Communication and Mobility Data".

The code compares individuals with social anxiety disorder (SAD) and healthy controls on passively sensed communication-app use, call behavior and mobility, and examines associations with symptom dimensions within the SAD group, using mixed-effects models that account for repeated weekly observations.

## Data

Participant data are not included. They contain GPS traces and communication metadata that could compromise participant privacy and are available on request from the corresponding author. The scripts expect the following files under `source/` in the repository root:

- `phone_applications_foreground/all_sensor_features.csv`, `phone_calls/all_sensor_features.csv` and `phone_locations_barnett/all_sensor_features.csv`: weekly weekday and weekend segment features extracted with RAPIDS.
- `demographics/demographics_clean.csv`: participant covariates (group, sex, age, years of education, occupation coded as a structured weekday schedule, enrollment during the COVID-19 distancing period, depressive comorbidity, enrollment date).
- `survey/LSAS_final.csv` and `survey/BFNE_final.csv`: biweekly LSAS-SR and BFNE scores.

## Environment

Python 3.11.9. Package versions are listed in `requirements.txt`.

```
pip install -r requirements.txt
```

## Run order

Run from the `analysis/` directory. Outputs are written to `analysis/output/`.

```
python run_01_build_datasets.py
python run_02_group_models.py
python run_03_symptom_models.py
python run_04_exploratory_models.py
python run_05_descriptives.py
```

| Script | Content |
|---|---|
| `config.py` | Paths, primary features, symptom scales, covariates and analysis constants |
| `datasets.py`, `run_01_build_datasets.py` | Eight-week window, location coverage filter, zero-filling of call weeks, matching of symptom assessments to segments |
| `stats.py` | Rank-based inverse-normal transformation, mixed-model and GEE fitting, contrasts, FDR correction |
| `run_02_group_models.py` | Group comparisons for the 14 primary features |
| `run_03_symptom_models.py` | Between-person symptom associations within the SAD group and the simultaneous-entry model of the fear and avoidance totals |
| `run_04_exploratory_models.py` | The same models for all 121 extracted features with full-set correction |
| `run_05_descriptives.py` | Group comparisons of demographics and baseline scales, test–retest reliability, descriptive statistics |

## Analysis overview

- **Transformation.** Volume features (counts, summed durations, distance traveled, time at home) are converted to per-day rates (weekday totals divided by five, weekend totals by two). Each feature is then rank-based inverse-normal transformed (Blom formula, average ranks) across the pooled analysis sample. Features with exact zeros in more than 40% of observations in either segment are dichotomized (nonzero versus zero) and analyzed with logistic generalized estimating equations (exchangeable working correlation, independence if it does not converge, robust standard errors).
- **Group comparisons.** Linear mixed models `y ~ group * segment + week + sex + age + education + occupation + distancing period` with a participant random intercept and random segment slope (restricted maximum likelihood). The primary group-difference estimate is the segment-averaged contrast; the group × segment interaction tests weekday–weekend moderation. p values use a t reference with degrees of freedom equal to the number of participants minus the number of person-level fixed effects.
- **Symptom associations.** Within the SAD group, each segment is matched to the earliest symptom assessment within 14 days of its start. Scores are decomposed into standardized between-person and within-person components entered jointly, with segment, week, covariates and depressive comorbidity as fixed effects, a participant random intercept and an assessment-occasion variance component.
- **Multiplicity.** Benjamini–Hochberg correction within the segment-averaged group contrasts (14 tests), the group × segment interactions (14 tests) and the between-person symptom coefficients (56 tests). Exploratory analyses apply the correction across all 121 features (121, 121 and 484 tests).
- The scripts also compute estimates that are not reported in the manuscript (segment-specific contrast family, within-person coefficients and sensitivity specifications of the group models).
