import numpy as np
import pandas as pd

from config import (
    CALL_ZERO_FILL_SUFFIXES, DEMOGRAPHICS_FILE, DISTANCING_END, LOCATION_COVERAGE_MIN,
    MATCH_WINDOW_DAYS, OBSERVATION_DAYS, SENSOR_FILES, SENSOR_PREFIX, SURVEY_FILES, WEEK_CENTER,
)


def load_demographics():
    demo = pd.read_csv(DEMOGRAPHICS_FILE, parse_dates=["dp_start", "dp_end"])
    keep = ["pid_str", "group", "sad", "male", "age", "edu_year",
            "structured_weekday_schedule", "covid_period", "dep_comorbid", "dp_start"]
    return demo[keep]


def load_sensor(sensor):
    prefix = SENSOR_PREFIX[sensor]
    df = pd.read_csv(SENSOR_FILES[sensor])
    df = df.rename(columns={c: c.removeprefix(prefix) for c in df.columns})
    df = df.rename(columns={"pid": "pid_str"})
    df["seg_start"] = pd.to_datetime(df["local_segment_start_datetime"])
    df["segment"] = (df["local_segment_label"] == "weekend").astype(int)
    assert not df.duplicated(["pid_str", "seg_start"]).any()
    assert df.loc[df.segment == 0, "seg_start"].dt.dayofweek.eq(0).all()
    assert df.loc[df.segment == 1, "seg_start"].dt.dayofweek.eq(5).all()
    drop = ["local_segment", "local_segment_label", "local_segment_start_datetime",
            "local_segment_end_datetime"]
    return df.drop(columns=drop)


def apply_location_coverage_filter(location):
    minutes = location["minutes_data_used"]
    threshold = location["segment"].map(LOCATION_COVERAGE_MIN)
    return location[minutes >= threshold].copy()


def compute_window_starts(sensors, demo):
    first_seen = (
        pd.concat([s[["pid_str", "seg_start"]] for s in sensors])
        .groupby("pid_str")["seg_start"].min()
    )
    win = demo.set_index("pid_str")["dp_start"]
    return win.fillna(first_seen).rename("win_start")


def zero_fill_calls(calls, apps):
    call_pids = calls["pid_str"].unique()
    app_cells = apps.loc[apps.pid_str.isin(call_pids), ["pid_str", "seg_start", "segment"]]
    observed = calls[["pid_str", "seg_start"]]
    fill_cells = app_cells.merge(observed, on=["pid_str", "seg_start"], how="left", indicator=True)
    fill_cells = fill_cells[fill_cells._merge == "left_only"].drop(columns="_merge")
    feature_cols = [c for c in calls.columns if c not in {"pid_str", "seg_start", "segment"}]
    filled = fill_cells.copy()
    for c in feature_cols:
        filled[c] = 0.0 if c.endswith(CALL_ZERO_FILL_SUFFIXES) else np.nan
    calls = calls.assign(zero_filled=0)
    filled = filled.assign(zero_filled=1)
    return pd.concat([calls, filled], ignore_index=True)


def truncate_to_window(df, win_start):
    df = df.merge(win_start, left_on="pid_str", right_index=True)
    end = df["win_start"] + pd.Timedelta(days=OBSERVATION_DAYS)
    df = df[(df.seg_start >= df.win_start) & (df.seg_start < end)].copy()
    df["week"] = (df.seg_start - df.win_start).dt.days // 7 + 1
    df["week_c"] = df["week"] - WEEK_CENTER
    df["distancing"] = (df.seg_start < pd.Timestamp(DISTANCING_END)).astype(int)
    return df.drop(columns="win_start")


def build_group_datasets():
    demo = load_demographics()
    raw = {name: load_sensor(name) for name in SENSOR_FILES}
    win_start = compute_window_starts(list(raw.values()), demo)
    datasets = {}
    datasets["apps"] = raw["apps"].assign(zero_filled=0)
    datasets["calls"] = zero_fill_calls(raw["calls"], raw["apps"])
    datasets["location"] = apply_location_coverage_filter(raw["location"]).assign(zero_filled=0)
    out = {}
    for name, df in datasets.items():
        df = truncate_to_window(df, win_start)
        if name == "location":
            km_per_day = df["disttravelled"] / df["segment"].map({0: 5, 1: 2}) / 1000
            df["gps_extreme"] = (km_per_day > 500).astype(int)
        out[name] = df.merge(demo.drop(columns="dp_start"), on="pid_str")
    return out


def load_surveys():
    surveys = {}
    for instrument, path in SURVEY_FILES.items():
        df = pd.read_csv(path, parse_dates=["datetime"])
        df["pid_str"] = "p" + df["pid"].astype(int).astype(str).str.zfill(3)
        surveys[instrument] = df.drop(columns=["pid", "group", "subjNum", "folder_name"])
    return surveys


def match_surveys(sensor_df, survey_df, score_cols):
    merged = sensor_df[["pid_str", "seg_start"]].reset_index().merge(
        survey_df[["pid_str", "datetime"] + score_cols], on="pid_str")
    in_window = (
        (merged.datetime >= merged.seg_start)
        & (merged.datetime < merged.seg_start + pd.Timedelta(days=MATCH_WINDOW_DAYS))
    )
    matches = (
        merged[in_window]
        .sort_values(["index", "datetime"], kind="stable")
        .drop_duplicates("index")
        .set_index("index")
    )
    result = sensor_df.join(matches[["datetime"] + score_cols], how="inner")
    return result.rename(columns={"datetime": "occ"})
