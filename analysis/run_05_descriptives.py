import json

import pandas as pd
from scipy import stats as sps

from config import (
    CORE_FEATURES, DATASETS_DIR, DEMOGRAPHICS_FILE, DIAGNOSTICS_DIR, SEGMENT_DAYS, SURVEY_FILES,
    TABLES_DIR, ensure_output_dirs,
)

SCALE_TOTALS = {"LSAS": "lsas_total", "BFNE": "bfne_total"}


def demographics_comparison():
    demo = pd.read_csv(DEMOGRAPHICS_FILE)
    sad, hc = demo[demo.group == "SAD"], demo[demo.group == "HC"]
    rows = []

    def cont(name, col):
        a, h = sad[col].dropna(), hc[col].dropna()
        rows.append({
            "characteristic": name,
            "sad": f"{a.mean():.2f} ({a.std():.2f})", "sad_n": len(a),
            "hc": f"{h.mean():.2f} ({h.std():.2f})", "hc_n": len(h),
            "p_student": sps.ttest_ind(a, h).pvalue,
            "p_welch": sps.ttest_ind(a, h, equal_var=False).pvalue})

    def binary(name, col, value=1):
        a, h = sad[col].dropna(), hc[col].dropna()
        table = [[(a == value).sum(), (a != value).sum()],
                 [(h == value).sum(), (h != value).sum()]]
        rows.append({
            "characteristic": name,
            "sad": f"{table[0][0]}/{len(a)}", "sad_n": len(a),
            "hc": f"{table[1][0]}/{len(h)}", "hc_n": len(h),
            "p_fisher": sps.fisher_exact(table).pvalue})

    binary("Male sex", "male")
    cont("Age, years", "age")
    cont("Education, years", "edu_year")
    binary("Structured weekday schedule", "structured_weekday_schedule")
    binary("Enrolled during distancing period", "covid_period")
    for name, col in [("Comorbid PDD/MDD", "dep_comorbid"), ("Comorbid PD", "pd_comorbid"),
                      ("Comorbid OCD", "ocd_comorbid"), ("Comorbid GAD", "gad_comorbid")]:
        rows.append({"characteristic": name, "sad": f"{int(sad[col].sum())}/{len(sad)}",
                     "sad_n": len(sad), "hc": "NA", "hc_n": 0})
    return pd.DataFrame(rows)


def baseline_and_reliability():
    baseline_rows, icc_rows = [], []
    demo = pd.read_csv(DEMOGRAPHICS_FILE)[["pid", "group"]]
    for instrument, path in SURVEY_FILES.items():
        total = SCALE_TOTALS[instrument]
        df = pd.read_csv(path, parse_dates=["datetime"]).drop(columns=["group"])
        df = df.merge(demo, on="pid")
        df = df.sort_values(["pid", "datetime"], kind="stable")
        first = df.drop_duplicates("pid", keep="first")
        a = first.loc[first.group == "SAD", total]
        h = first.loc[first.group == "HC", total]
        welch = sps.ttest_ind(a, h, equal_var=False)
        baseline_rows.append({
            "scale": instrument,
            "sad": f"{a.mean():.2f} ({a.std():.2f})", "sad_n": len(a),
            "hc": f"{h.mean():.2f} ({h.std():.2f})", "hc_n": len(h),
            "welch_t": welch.statistic, "p_welch": welch.pvalue})
        counts = df.groupby("pid").size()
        repeated = counts[counts >= 2].index
        for group_name in ("SAD", "HC"):
            sub = df[(df.group == group_name) & df.pid.isin(repeated)]
            pairs = pd.DataFrame({
                "first": sub.drop_duplicates("pid", keep="first").set_index("pid")[total],
                "last": sub.drop_duplicates("pid", keep="last").set_index("pid")[total]})
            icc_rows.append({"scale": instrument, "group": group_name,
                             "n": len(pairs), "icc_1_1": icc_1_1(pairs)})
    return pd.DataFrame(baseline_rows), pd.DataFrame(icc_rows)


def icc_1_1(pairs):
    values = pairs.to_numpy(dtype=float)
    n, k = values.shape
    grand = values.mean()
    ms_between = k * ((values.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_within = ((values - values.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)


def core_descriptives():
    rows = []
    for sensor, column, label in CORE_FEATURES:
        df = pd.read_csv(DATASETS_DIR / f"group_{sensor}.csv")
        for seg, seg_name in ((0, "weekday"), (1, "weekend")):
            d = df[df.segment == seg].dropna(subset=[column])
            for group_name in ("SAD", "HC"):
                s = d.loc[d.group == group_name, column]
                rows.append({
                    "feature": column, "label": label, "segment": seg_name,
                    "group": group_name, "n_rows": len(s),
                    "n_participants": d.loc[d.group == group_name, "pid_str"].nunique(),
                    "median": s.median(), "q1": s.quantile(0.25), "q3": s.quantile(0.75),
                    "share_zero": (s == 0).mean()})
    return pd.DataFrame(rows)


def sample_accounting():
    with open(DATASETS_DIR / "build_report.json") as f:
        report = json.load(f)
    rows = []
    for sensor, info in report.items():
        rows.append({"sensor": sensor, **{k: v for k, v in info.items()
                                          if not isinstance(v, dict)},
                     **{f"rows_{k}": v for k, v in info["rows_by_group"].items()},
                     **{f"participants_{k}": v for k, v in info["participants_by_group"].items()}})
    return pd.DataFrame(rows)


def gps_outliers(threshold_km_per_day=500):
    df = pd.read_csv(DATASETS_DIR / "group_location.csv")
    km_per_day = df["disttravelled"] / df["segment"].map(SEGMENT_DAYS) / 1000
    out = df.loc[km_per_day > threshold_km_per_day,
                 ["pid_str", "group", "seg_start", "segment", "disttravelled"]]
    return out.assign(km_per_day=km_per_day[out.index].round(1))


def main():
    ensure_output_dirs()
    demographics_comparison().to_csv(TABLES_DIR / "demographics_comparison.csv", index=False)
    baseline, icc = baseline_and_reliability()
    baseline.to_csv(TABLES_DIR / "baseline_scales.csv", index=False)
    icc.to_csv(TABLES_DIR / "reliability_icc.csv", index=False)
    core_descriptives().to_csv(TABLES_DIR / "core_descriptives.csv", index=False)
    sample_accounting().to_csv(TABLES_DIR / "sample_accounting.csv", index=False)
    gps_outliers().to_csv(DIAGNOSTICS_DIR / "gps_outliers.csv", index=False)
    print(baseline.round(4).to_string(index=False))
    print(icc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
