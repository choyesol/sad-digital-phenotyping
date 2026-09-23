import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


def int_transform(x):
    x = pd.Series(x, dtype=float)
    ranks = x.rank(method="average")
    n = x.notna().sum()
    return pd.Series(sps.norm.ppf((ranks - 3 / 8) / (n + 1 / 4)), index=x.index)


def t_pvalue(estimate, se, df):
    tval = estimate / se
    return 2 * sps.t.sf(np.abs(tval), df)


def t_ci(estimate, se, df, level=0.95):
    q = sps.t.ppf(1 - (1 - level) / 2, df)
    return estimate - q * se, estimate + q * se


def person_level_df(data, result, group_col="pid_str"):
    exog = pd.DataFrame(result.model.exog, columns=result.model.exog_names)
    exog[group_col] = data[group_col].to_numpy()
    within_var = exog.groupby(group_col).var(ddof=0).max()
    k_person = int((within_var < 1e-12).sum())
    return data[group_col].nunique() - k_person


def fit_mixedlm(data, formula, group_col="pid_str", re_formula=None, vc_formula=None):
    model = smf.mixedlm(formula, data, groups=data[group_col],
                        re_formula=re_formula, vc_formula=vc_formula)
    for kwargs in ({}, {"method": "powell"}, {"method": "cg"}):
        try:
            result = model.fit(reml=True, **kwargs)
        except Exception:
            continue
        if result.converged:
            return result, kwargs.get("method", "default")
    return None, "failed"


def fit_gee_logit(data, formula, group_col="pid_str"):
    for name, cov_struct in (("exchangeable", sm.cov_struct.Exchangeable()),
                             ("independence", sm.cov_struct.Independence())):
        model = smf.gee(formula, groups=data[group_col], data=data,
                        family=sm.families.Binomial(), cov_struct=cov_struct)
        try:
            result = model.fit(maxiter=200)
        except Exception:
            continue
        if result.converged and not result.params.isna().any():
            return result, name
    return None, "failed"


def contrast(result, weights):
    params = result.fe_params if hasattr(result, "fe_params") else result.params
    names = list(params.index)
    L = np.zeros((1, len(names)))
    for name, w in weights.items():
        L[0, names.index(name)] = w
    tt = result.t_test(L)
    return float(np.ravel(tt.effect)[0]), float(np.ravel(tt.sd)[0])


def nakagawa_r2(result, data, re_formula=None):
    fe_pred = np.asarray(result.model.exog) @ np.asarray(result.fe_params)
    var_f = np.var(fe_pred, ddof=1)
    G = np.asarray(result.cov_re)
    if re_formula == "~segment":
        z = np.column_stack([np.ones(len(data)), data["segment"].to_numpy()])
        var_re = float(np.mean(np.einsum("ij,jk,ik->i", z, G, z)))
    else:
        var_re = float(G[0, 0]) if G.size else 0.0
    if len(result.vcomp):
        var_re += float(np.sum(result.vcomp))
    var_e = float(result.scale)
    total = var_f + var_re + var_e
    return var_f / total, (var_f + var_re) / total


def residual_summary(result, data):
    resid = np.asarray(result.resid)
    seg = data["segment"].to_numpy()
    return {
        "resid_skew": sps.skew(resid),
        "resid_kurtosis": sps.kurtosis(resid),
        "resid_sd_weekday": resid[seg == 0].std(),
        "resid_sd_weekend": resid[seg == 1].std(),
    }


def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    if mask.sum():
        adjusted[mask] = multipletests(p[mask], method="fdr_bh")[1]
    return adjusted


def add_between_within(data, score_col, occ_col="occ", group_col="pid_str"):
    occ = data.groupby([group_col, occ_col])[score_col].first().reset_index()
    person_mean = occ.groupby(group_col)[score_col].mean()
    sd_between = person_mean.std(ddof=1)
    deviations = occ[score_col] - occ[group_col].map(person_mean)
    within_sds = deviations.groupby(occ[group_col]).std(ddof=1).dropna()
    sd_within = float(np.sqrt(np.mean(within_sds**2))) if len(within_sds) else np.nan
    out = data.copy()
    out["z_between"] = (out[group_col].map(person_mean) - person_mean.mean()) / sd_between
    if sd_within and np.isfinite(sd_within) and sd_within > 0:
        out["z_within"] = (out[score_col] - out[group_col].map(person_mean)) / sd_within
    else:
        out["z_within"] = 0.0
    return out
