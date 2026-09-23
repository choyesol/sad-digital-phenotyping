import json

import pandas as pd

from config import DATASETS_DIR, SYMPTOM_SCALES, ensure_output_dirs
from datasets import build_group_datasets, load_surveys, match_surveys

LSAS_COLS = [c for i, c, _ in SYMPTOM_SCALES if i == "LSAS"]
BFNE_COLS = [c for i, c, _ in SYMPTOM_SCALES if i == "BFNE"]


def main():
    ensure_output_dirs()
    datasets = build_group_datasets()
    surveys = load_surveys()
    report = {}
    for name, df in datasets.items():
        df.to_csv(DATASETS_DIR / f"group_{name}.csv", index=False)
        report[name] = {
            "rows": len(df),
            "participants": df.pid_str.nunique(),
            "rows_by_group": df.group.value_counts().to_dict(),
            "participants_by_group": df.groupby("group").pid_str.nunique().to_dict(),
            "zero_filled_rows": int(df.zero_filled.sum()),
            "weeks_per_participant_median": float(df.groupby("pid_str").week.nunique().median()),
        }
        sad = df[df.group == "SAD"]
        for instrument, cols in (("LSAS", LSAS_COLS), ("BFNE", BFNE_COLS)):
            matched = match_surveys(sad, surveys[instrument], cols)
            matched.to_csv(DATASETS_DIR / f"symptom_{name}_{instrument}.csv", index=False)
            report[name][f"symptom_rows_{instrument}"] = len(matched)
            report[name][f"symptom_participants_{instrument}"] = matched.pid_str.nunique()
            report[name][f"symptom_occasions_{instrument}"] = int(
                matched.groupby(["pid_str", "occ"]).ngroups)
    with open(DATASETS_DIR / "build_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
