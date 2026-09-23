"""Project-wide constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CURATED = ROOT / "data" / "curated"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

DATASET_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.zip"

BARCA_ID = 131            # FC Barcelona on Transfermarkt
LALIGA = "ES1"
TOP5 = ["GB1", "ES1", "IT1", "L1", "FR1"]   # PL, LaLiga, Serie A, Bundesliga, Ligue 1
FIRST_SEASON = 2010       # 2010/11
BASE_SEASON = 2025        # euros expressed in 2025/26 football-money terms

MIN_PEER_MINUTES = 450    # La Liga peers need this many minutes to set the benchmark
SHRINK_MINUTES = 900      # per-90 z-scores shrink toward 0 below ~10 full games

# Position groups used for role-aware scoring
SUBPOS_TO_GROUP = {
    "Goalkeeper": "GK",
    "Centre-Back": "CB",
    "Left-Back": "FB", "Right-Back": "FB",
    "Defensive Midfield": "MID", "Central Midfield": "MID",
    "Left Midfield": "MID", "Right Midfield": "MID",
    "Attacking Midfield": "AM",
    "Left Winger": "AM", "Right Winger": "AM",
    "Centre-Forward": "FW", "Second Striker": "FW",
}
POSITION_TO_GROUP = {"Goalkeeper": "GK", "Defender": "CB", "Midfield": "MID", "Attack": "FW"}

# How much each metric counts, by role. Keys are z-scored metrics built in performance.py.
#   g90 = goals/90, a90 = assists/90, mins = share of the team's league minutes,
#   onoff = team points-per-game with him (60+ min) minus without,
#   cs = team goals conceded per game when he plays 60+ (sign flipped: fewer = better)
ROLE_WEIGHTS = {
    "GK":  {"g90": 0.00, "a90": 0.00, "mins": 0.45, "onoff": 0.20, "cs": 0.35},
    "CB":  {"g90": 0.05, "a90": 0.00, "mins": 0.40, "onoff": 0.25, "cs": 0.30},
    "FB":  {"g90": 0.05, "a90": 0.25, "mins": 0.30, "onoff": 0.20, "cs": 0.20},
    "MID": {"g90": 0.15, "a90": 0.25, "mins": 0.35, "onoff": 0.25, "cs": 0.00},
    "AM":  {"g90": 0.30, "a90": 0.30, "mins": 0.20, "onoff": 0.20, "cs": 0.00},
    "FW":  {"g90": 0.45, "a90": 0.20, "mins": 0.20, "onoff": 0.15, "cs": 0.00},
}

# Trophy points: extra credit for goals, assists and minutes in a competition the club WON.
# Relative weights set by preference (UCL > domestic league > other cups); the market model learns the euro value.
TROPHY_WEIGHTS = {"CL": 3.0, "ES1": 2.0, "EL": 1.0, "CDR": 1.0, "KLUB": 0.75, "SUC": 0.5, "USC": 0.5}
TROPHY_NAMES = {"CL": "Champions League", "ES1": "La Liga", "EL": "Europa League", "CDR": "Copa del Rey",
                "KLUB": "Club World Cup", "SUC": "Supercopa", "USC": "UEFA Super Cup"}
# the same weights by competition type, for every other league and cup in the dataset
TROPHY_TYPE_WEIGHTS = {"domestic_league": 2.0, "domestic_cup": 1.0, "domestic_super_cup": 0.5,
                       "uefa_conference_league": 0.5}
# leagues whose champion is decided by play-offs, so "most points" would be wrong: no league title counted
PLAYOFF_LEAGUES = {"MLS1", "MEX1", "ARG1", "AUS1", "BE1"}
