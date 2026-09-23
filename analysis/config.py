from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
OUTPUT = Path(__file__).resolve().parent / "output"
DATASETS_DIR = OUTPUT / "datasets"
TABLES_DIR = OUTPUT / "tables"
DIAGNOSTICS_DIR = OUTPUT / "diagnostics"

SENSOR_FILES = {
    "apps": SOURCE / "phone_applications_foreground" / "all_sensor_features.csv",
    "calls": SOURCE / "phone_calls" / "all_sensor_features.csv",
    "location": SOURCE / "phone_locations_barnett" / "all_sensor_features.csv",
}
SENSOR_PREFIX = {
    "apps": "phone_applications_foreground_rapids_",
    "calls": "phone_calls_rapids_",
    "location": "phone_locations_barnett_",
}
DEMOGRAPHICS_FILE = SOURCE / "demographics" / "demographics_clean.csv"
SURVEY_FILES = {
    "LSAS": SOURCE / "survey" / "LSAS_final.csv",
    "BFNE": SOURCE / "survey" / "BFNE_final.csv",
}

OBSERVATION_DAYS = 56
SEGMENT_DAYS = {0: 5, 1: 2}
LOCATION_COVERAGE_MIN = {0: 5040, 1: 2016}
DISTANCING_END = "2022-04-18"
MATCH_WINDOW_DAYS = 14
ZERO_BINARIZE_THRESHOLD = 0.40
WEEK_CENTER = 4.5

COVARIATES = ["male", "age", "edu_year", "structured_weekday_schedule", "covid_period"]

CORE_FEATURES = [
    ("apps", "counteventratio_chat", "Chat app launch proportion"),
    ("apps", "counteventratio_sns", "SNS app launch proportion"),
    ("apps", "counteventratio_community", "Community app launch proportion"),
    ("apps", "counteventratio_work", "Work app launch proportion"),
    ("calls", "outgoing_count", "Outgoing call count"),
    ("calls", "incoming_count", "Incoming call count"),
    ("calls", "outgoing_sumduration", "Total outgoing call duration"),
    ("calls", "incoming_sumduration", "Total incoming call duration"),
    ("location", "disttravelled", "Distance traveled"),
    ("location", "rog", "Radius of gyration"),
    ("location", "hometime", "Time spent at home"),
    ("location", "siglocentropy", "Location entropy"),
    ("location", "probpause", "Pause ratio"),
    ("location", "circdnrtn", "Location regularity (circadian routine)"),
]

SYMPTOM_SCALES = [
    ("LSAS", "lsas_total", "LSAS-SR total"),
    ("LSAS", "lsas_fear_total", "LSAS-SR fear total"),
    ("LSAS", "lsas_avoid_total", "LSAS-SR avoidance total"),
    ("BFNE", "bfne_total", "BFNE"),
]

SPECIFICITY_TESTS = [
    ("apps", "counteventratio_chat", "lsas_fear_total", "lsas_avoid_total"),
]

CALL_ZERO_FILL_SUFFIXES = ("_count", "_distinctcontacts", "_sumduration", "_countmostfrequentcontact")

EXCLUDED_EXPLORATORY = {
    "minutes_data_used",
    "countepisodeall", "countepisodechat_custom", "countepisodesns_custom",
    "countepisodecommunity_custom", "countepisodework_custom",
    "countepisodecommunication_custom_category", "countepisodetop1global",
}


def is_per_day(sensor, column):
    if sensor == "calls":
        return column.endswith(CALL_ZERO_FILL_SUFFIXES)
    if sensor == "apps":
        return (column.startswith("countevent") and "ratio" not in column) or \
            column.startswith("countepisode") or column.startswith("sumduration")
    if sensor == "location":
        return column in {"hometime", "disttravelled"}
    return False


def ensure_output_dirs():
    for d in (DATASETS_DIR, TABLES_DIR, DIAGNOSTICS_DIR):
        d.mkdir(parents=True, exist_ok=True)
