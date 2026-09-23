import numpy as np
import pandas as pd

from config import (
    CORE_FEATURES, COVARIATES, DATASETS_DIR, EXCLUDED_EXPLORATORY, SEGMENT_DAYS, SYMPTOM_SCALES,
    TABLES_DIR, ZERO_BINARIZE_THRESHOLD, ensure_output_dirs, is_per_day,
)
from run_02_group_models import fit_spec
from run_03_symptom_models import build_outcome, fit_symptom, prepare, term_row
from stats import add_between_within, bh_adjust

CORE_SET = {(s, c) for s, c, _ in CORE_FEATURES}
META_COLS = {"pid_str", "group", "sad", "segment", "week", "week_c", "seg_start", "occ",
             "zero_filled", "distancing", "gps_extreme", "male", "age", "edu_year",
             "structured_weekday_schedule", "covid_period", "dep_comorbid"}


def feature_roster(sensor):
    df = pd.read_csv(DATASETS_DIR / f"group_{sensor}.csv", nrows=200)
    scales = {c for _, c, _ in SYMPTOM_SCALES}
    cols = []
    for c in df.columns:
        if c in META_COLS or c in EXCLUDED_EXPLORATORY or c in scales:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def estimable(df):
    if df.pid_str.nunique() < 20 or len(df) < 100:
        return False
    by_seg = df.groupby("segment")["y_input"]
    return bool((by_seg.std() > 0).all() and (by_seg.apply(lambda s: (s == 0).mean()) < 0.95).any())


def group_models():
    rows = []
    for sensor in ("apps", "calls", "location"):
        data = pd.read_csv(DATASETS_DIR / f"group_{sensor}.csv")
        for column in feature_roster(sensor):
            df = data[["pid_str", "sad", "segment", "week_c", "zero_filled", "distancing",
                       column] + COVARIATES].dropna(subset=[column] + COVARIATES).copy()
            divisor = df["segment"].map(SEGMENT_DAYS) if is_per_day(sensor, column) else 1.0
            df["y_input"] = df[column] / divisor
            if not estimable(df):
                continue
            shares = df.groupby("segment")["y_input"].apply(lambda s: (s == 0).mean())
            binarize = bool((shares > ZERO_BINARIZE_THRESHOLD).any())
            fit = fit_spec(df, binarize)
            if fit is None:
                continue
            rows.append({"sensor": sensor, "feature": column, "binarized": int(binarize),
                         "is_core": int((sensor, column) in CORE_SET), **fit[0]})
            print(f"group {sensor}:{column}", flush=True)
    out = pd.DataFrame(rows)
    out["padj_fullset_avg"] = bh_adjust(out.p_avg)
    out["padj_fullset_int"] = bh_adjust(out.p_int)
    adj = bh_adjust(np.concatenate([out.p_wd, out.p_we]))
    out["padj_fullset_wd_secondary"] = adj[: len(out)]
    out["padj_fullset_we_secondary"] = adj[len(out):]
    out.to_csv(TABLES_DIR / "exploratory_group.csv", index=False)
    return out


def symptom_models():
    between_rows, within_rows = [], []
    for sensor in ("apps", "calls", "location"):
        roster = feature_roster(sensor)
        for instrument, scale, scale_label in SYMPTOM_SCALES:
            data = prepare(sensor, [scale])
            for column in roster:
                if column not in data.columns:
                    continue
                df = build_outcome(data, sensor, column)
                if not estimable(df):
                    continue
                df = add_between_within(df, scale)
                shares = df.groupby("segment")["y_input"].apply(lambda s: (s == 0).mean())
                binarize = bool((shares > ZERO_BINARIZE_THRESHOLD).any())
                result, df_t, extra = fit_symptom(df, ["z_between", "z_within"], binarize)
                if result is None:
                    continue
                base = {"sensor": sensor, "feature": column, "scale": scale,
                        "binarized": int(binarize), "n_rows": len(df),
                        "n_participants": df.pid_str.nunique(), "model": extra["model"],
                        "is_core": int((sensor, column) in CORE_SET)}
                between_rows.append({**base, **term_row(result, df_t, "z_between")})
                within_rows.append({**base, **term_row(result, df_t, "z_within")})
            print(f"symptom {sensor}:{scale}", flush=True)
    between = pd.DataFrame(between_rows)
    within = pd.DataFrame(within_rows)
    between["padj_fullset"] = bh_adjust(between.p)
    within["padj_fullset"] = bh_adjust(within.p)
    between.to_csv(TABLES_DIR / "exploratory_symptom_between.csv", index=False)
    within.to_csv(TABLES_DIR / "exploratory_symptom_within.csv", index=False)
    return between


def main():
    ensure_output_dirs()
    group = group_models()
    between = symptom_models()
    core_group = group[group.is_core == 1][
        ["sensor", "feature", "p_avg", "padj_fullset_avg", "p_int", "padj_fullset_int",
         "p_wd", "padj_fullset_wd_secondary", "p_we", "padj_fullset_we_secondary"]]
    core_symptom = between[between.is_core == 1][
        ["sensor", "feature", "scale", "p", "padj_fullset"]]
    core_group.to_csv(TABLES_DIR / "fullset_check_group.csv", index=False)
    core_symptom.to_csv(TABLES_DIR / "fullset_check_symptom.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
