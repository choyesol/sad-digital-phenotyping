import numpy as np
import pandas as pd

from config import (
    CORE_FEATURES, COVARIATES, DATASETS_DIR, SEGMENT_DAYS, SPECIFICITY_TESTS, SYMPTOM_SCALES,
    TABLES_DIR, ZERO_BINARIZE_THRESHOLD, ensure_output_dirs, is_per_day,
)
from stats import (
    add_between_within, bh_adjust, contrast, fit_gee_logit, fit_mixedlm, int_transform,
    person_level_df, residual_summary, t_ci, t_pvalue,
)

SYMPTOM_COVARIATES = COVARIATES + ["dep_comorbid"]
FIXED_TAIL = "segment + week_c + " + " + ".join(SYMPTOM_COVARIATES)
SCALE_INSTRUMENT = {c: i for i, c, _ in SYMPTOM_SCALES}


def prepare(sensor, scale_cols):
    instrument = SCALE_INSTRUMENT[scale_cols[0]]
    df = pd.read_csv(DATASETS_DIR / f"symptom_{sensor}_{instrument}.csv")
    df = df.dropna(subset=SYMPTOM_COVARIATES + scale_cols).copy()
    return df


def build_outcome(df, sensor, column):
    df = df.dropna(subset=[column]).copy()
    divisor = df["segment"].map(SEGMENT_DAYS) if is_per_day(sensor, column) else 1.0
    df["y_input"] = df[column] / divisor
    return df


def is_binarized(df):
    shares = df.groupby("segment")["y_input"].apply(lambda s: (s == 0).mean())
    return bool((shares > ZERO_BINARIZE_THRESHOLD).any())


def fit_symptom(df, predictors, binarize):
    formula_rhs = " + ".join(predictors) + " + " + FIXED_TAIL
    if binarize:
        df = df.assign(y=(df["y_input"] > 0).astype(int))
        result, struct = fit_gee_logit(df, f"y ~ {formula_rhs}")
        if result is None:
            return None, None, None
        df_t = df.pid_str.nunique() - (len(predictors) // 2 + 1 + len(SYMPTOM_COVARIATES))
        return result, df_t, {"model": f"gee_logit_{struct}", "occ_var": np.nan}
    df = df.assign(y=int_transform(df["y_input"]))
    result, method = fit_mixedlm(df, f"y ~ {formula_rhs}", re_formula="1",
                                 vc_formula={"occ": "0 + C(occ)"})
    spec = "ri+occ"
    if result is None:
        result, method = fit_mixedlm(df, f"y ~ {formula_rhs}", re_formula="1")
        spec = "ri_only"
    if result is None:
        return None, None, None
    df_t = person_level_df(df, result)
    occ_var = float(np.sum(result.vcomp)) if len(result.vcomp) else 0.0
    extra = {"model": f"lmm_{spec}", "opt": method, "occ_var": occ_var,
             "pid_var": float(result.cov_re.iloc[0, 0]), "resid_var": float(result.scale),
             **residual_summary(result, df)}
    return result, df_t, extra


def term_row(result, df_t, term):
    b = result.params[term] if not hasattr(result, "fe_params") else result.fe_params[term]
    se = result.bse[term]
    lo, hi = t_ci(b, se, df_t)
    return {"b": b, "se": se, "lo": lo, "hi": hi, "p": t_pvalue(b, se, df_t)}


def main():
    ensure_output_dirs()
    between_rows, within_rows = [], []
    for sensor, column, label in CORE_FEATURES:
        for instrument, scale, scale_label in SYMPTOM_SCALES:
            df = prepare(sensor, [scale])
            df = build_outcome(df, sensor, column)
            df = add_between_within(df, scale)
            binarize = is_binarized(df)
            result, df_t, extra = fit_symptom(df, ["z_between", "z_within"], binarize)
            if result is None:
                continue
            base = {"sensor": sensor, "feature": column, "label": label,
                    "scale": scale, "scale_label": scale_label,
                    "binarized": int(binarize), "n_rows": len(df),
                    "n_participants": df.pid_str.nunique(), "df_t": df_t,
                    "n_occasions": int(df.groupby(["pid_str", "occ"]).ngroups), **extra}
            between_rows.append({**base, **term_row(result, df_t, "z_between")})
            within_rows.append({**base, **term_row(result, df_t, "z_within")})

    between = pd.DataFrame(between_rows)
    within = pd.DataFrame(within_rows)
    assert len(between) == len(CORE_FEATURES) * len(SYMPTOM_SCALES)
    assert len(within) == len(CORE_FEATURES) * len(SYMPTOM_SCALES)
    between["padj"] = bh_adjust(between.p)
    within["padj"] = bh_adjust(within.p)
    between.to_csv(TABLES_DIR / "symptom_between.csv", index=False)
    within.to_csv(TABLES_DIR / "symptom_within.csv", index=False)

    spec_rows = []
    for sensor, column, dim1, dim2 in SPECIFICITY_TESTS:
        df = prepare(sensor, [dim1, dim2])
        df = build_outcome(df, sensor, column)
        df = add_between_within(df, dim1).rename(
            columns={"z_between": "dim1_between", "z_within": "dim1_within"})
        df = add_between_within(df, dim2).rename(
            columns={"z_between": "dim2_between", "z_within": "dim2_within"})
        predictors = ["dim1_between", "dim2_between", "dim1_within", "dim2_within"]
        binarize = is_binarized(df)
        result, df_t, extra = fit_symptom(df, predictors, binarize)
        if result is None:
            continue
        person = df.groupby("pid_str")[["dim1_between", "dim2_between"]].first()
        r_between = person.corr().iloc[0, 1]
        diff, diff_se = contrast(result, {"dim1_between": 1, "dim2_between": -1})
        lo, hi = t_ci(diff, diff_se, df_t)
        spec_rows.append({
            "sensor": sensor, "feature": column, "dim1": dim1, "dim2": dim2,
            "n_rows": len(df), "n_participants": df.pid_str.nunique(), "df_t": df_t,
            "r_between_dims": r_between, "vif_between": 1 / (1 - r_between**2),
            **{f"dim1_{k}": v for k, v in term_row(result, df_t, "dim1_between").items()},
            **{f"dim2_{k}": v for k, v in term_row(result, df_t, "dim2_between").items()},
            "diff_b": diff, "diff_se": diff_se, "diff_lo": lo, "diff_hi": hi,
            "diff_p": t_pvalue(diff, diff_se, df_t), **extra})
    pd.DataFrame(spec_rows).to_csv(TABLES_DIR / "specificity.csv", index=False)

    heat = between.pivot(index="label", columns="scale_label", values="b")
    order = [lab for _, _, lab in SYMPTOM_SCALES]
    labels = [lab for _, _, lab in CORE_FEATURES]
    print(heat.reindex(labels)[order].round(2).to_string())
    sig = between[between.padj < 0.05]
    print(f"\nbetween-person associations with FDR p<.05: {len(sig)}")
    print(sig[["label", "scale_label", "b", "lo", "hi", "p", "padj"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
