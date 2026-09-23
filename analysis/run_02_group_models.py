import numpy as np
import pandas as pd

from config import (
    CORE_FEATURES, COVARIATES, DATASETS_DIR, SEGMENT_DAYS, TABLES_DIR,
    ZERO_BINARIZE_THRESHOLD, ensure_output_dirs, is_per_day,
)
from stats import (
    bh_adjust, contrast, fit_gee_logit, fit_mixedlm, int_transform, nakagawa_r2,
    person_level_df, residual_summary, t_ci, t_pvalue,
)

BASE_TERMS = "sad*segment + week_c"
FULL_FIXED = f"{BASE_TERMS} + " + " + ".join(COVARIATES)

SENSITIVITY_SPECS = {
    "ri_only": {"re_formula": "1"},
    "per_segment_int": {"transform": "per_segment"},
    "no_zero_fill": {"drop_zero_filled": True},
    "no_schedule": {"fixed": FULL_FIXED.replace(" + structured_weekday_schedule", ""),
                    "covariates": [c for c in COVARIATES if c != "structured_weekday_schedule"]},
    "schedule_x_segment": {"fixed": FULL_FIXED + " + structured_weekday_schedule:segment"},
    "covid_timevarying": {"fixed": FULL_FIXED.replace("covid_period", "distancing")},
    "gps_trim": {"drop_gps_extreme": True},
}


def prepare(sensor, column, covariates):
    df = pd.read_csv(DATASETS_DIR / f"group_{sensor}.csv")
    keep = ["pid_str", "sad", "segment", "week_c", "zero_filled", "distancing", column]
    if "gps_extreme" in df.columns:
        keep.append("gps_extreme")
    df = df[keep + COVARIATES].dropna(subset=[column] + covariates).copy()
    divisor = df["segment"].map(SEGMENT_DAYS) if is_per_day(sensor, column) else 1.0
    df["y_input"] = df[column] / divisor
    return df


def effect_rows(result, df_t):
    b_we, se_we = contrast(result, {"sad": 1, "sad:segment": 1})
    b_avg, se_avg = contrast(result, {"sad": 1, "sad:segment": 0.5})
    out = {"df_t": df_t}
    for tag, b, se in (("avg", b_avg, se_avg),
                       ("wd", result.params["sad"], result.bse["sad"]),
                       ("we", b_we, se_we),
                       ("int", result.params["sad:segment"], result.bse["sad:segment"])):
        lo, hi = t_ci(b, se, df_t)
        out.update({f"b_{tag}": b, f"se_{tag}": se, f"lo_{tag}": lo, f"hi_{tag}": hi,
                    f"p_{tag}": t_pvalue(b, se, df_t)})
    return out


def fit_spec(df, binarize, fixed=FULL_FIXED, re_formula="~segment", transform="pooled",
             drop_zero_filled=False, drop_gps_extreme=False, covariates=COVARIATES):
    df = df[df.zero_filled == 0].copy() if drop_zero_filled else df.copy()
    if drop_gps_extreme:
        df = df[df.gps_extreme == 0].copy()
    n_pid = df.pid_str.nunique()
    if binarize:
        df["y"] = (df["y_input"] > 0).astype(int)
        result, struct = fit_gee_logit(df, f"y ~ {fixed}")
        if result is None:
            return None
        row = {"model": "gee_logit", "re_spec": struct, "opt": "gee"}
        df_t = n_pid - (2 + len(covariates))
    else:
        if transform == "per_segment":
            df["y"] = df.groupby("segment")["y_input"].transform(int_transform)
        else:
            df["y"] = int_transform(df["y_input"])
        result, method = fit_mixedlm(df, f"y ~ {fixed}", re_formula=re_formula)
        used = re_formula
        if result is None and re_formula != "1":
            result, method = fit_mixedlm(df, f"y ~ {fixed}", re_formula="1")
            used = "1"
        if result is None:
            return None
        df_t = person_level_df(df, result)
        r2m, r2c = nakagawa_r2(result, df, used)
        row = {"model": "lmm", "re_spec": used, "opt": method,
               "r2_marginal": r2m, "r2_conditional": r2c, **residual_summary(result, df)}
    row.update({"n_rows": len(df), "n_participants": n_pid, **effect_rows(result, df_t)})
    return row, result, df


def main():
    ensure_output_dirs()
    primary_rows, sensitivity_rows = [], []
    for sensor, column, label in CORE_FEATURES:
        df = prepare(sensor, column, COVARIATES)
        shares = df.groupby("segment")["y_input"].apply(lambda s: (s == 0).mean()).reindex([0, 1])
        binarize = bool((shares > ZERO_BINARIZE_THRESHOLD).any())
        base = {"sensor": sensor, "feature": column, "label": label,
                "zero_share_wd": shares[0], "zero_share_we": shares[1],
                "binarized": int(binarize)}
        row, _, _ = fit_spec(df, binarize)
        primary_rows.append({**base, **row})
        for spec_name, spec in SENSITIVITY_SPECS.items():
            if spec_name == "no_zero_fill" and (sensor != "calls" or df.zero_filled.sum() == 0):
                continue
            if spec_name == "gps_trim" and (sensor != "location" or df.gps_extreme.sum() == 0):
                continue
            if binarize and spec_name in ("ri_only", "per_segment_int", "no_zero_fill", "gps_trim"):
                continue
            spec_df = prepare(sensor, column, spec["covariates"]) if "covariates" in spec else df
            kwargs = {k: v for k, v in spec.items() if k != "covariates"}
            fit = fit_spec(spec_df, binarize, **kwargs)
            if fit:
                sensitivity_rows.append({**base, "spec": spec_name, **fit[0]})

    primary = pd.DataFrame(primary_rows)
    primary["padj_avg"] = bh_adjust(primary.p_avg)
    primary["padj_int"] = bh_adjust(primary.p_int)
    contrast_adj = bh_adjust(np.concatenate([primary.p_wd, primary.p_we]))
    primary["padj_wd_secondary"] = contrast_adj[: len(primary)]
    primary["padj_we_secondary"] = contrast_adj[len(primary):]
    primary.to_csv(TABLES_DIR / "group_primary.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(TABLES_DIR / "group_sensitivity.csv", index=False)
    cols = ["label", "binarized", "b_avg", "padj_avg", "b_wd", "b_we", "b_int", "padj_int",
            "re_spec"]
    print(primary[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
