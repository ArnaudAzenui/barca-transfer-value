"""Market model: what La Liga's market pays for a given season of output.

Two versions are fitted:
  * La Liga only (~7,600 player-seasons): values each season a player spent AT the club (the outcome).
  * All 14 leagues (~92,000 player-seasons, with league fixed effects so a +1 z-score in the Eredivisie is worth less
    than in La Liga): values a player's seasons BEFORE he signed, wherever he played (the decision).
    log(end-of-season market value, inflation-adjusted) ~ role x league z-scores + age + age^2 + team strength
                                                          + role x European goals/assists z + European minutes z
                                                          + whether his club was in the Champions / Europa League
This is the "predict value from stats" linear regression, run at league scale, so that each Barça season can be
translated into euros: the Performance-Implied Value (PIV) of that season.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from .data import load, season_of
from .inflation import adjuster

FORMULA = ("log_mv ~ C(league) + C(group) + C(group):(z_g90 + z_a90 + z_mins + z_onoff + z_cs) "
           "+ age + I(age**2) + team_ppg "
           # European matches: role-specific goal/assist output + how much he was used + the stage itself
           "+ " + " + ".join(f"eu_g90_{g} + eu_a90_{g}" for g in ["CB", "FB", "MID", "AM", "FW"])
           + " + eu_z_mins + in_cl + in_el"
           # trophies: points for playing/scoring in competitions the club won (CL 3 > La Liga 2 > cups)
           + " + trophy_pts")
EU_ROLES = ["CB", "FB", "MID", "AM", "FW"]   # goalkeepers: European minutes only


def end_of_season_values(index_df) -> pd.DataFrame:
    v = load("player_valuations")[["player_id", "date", "market_value_in_eur"]].copy()
    v = v[v["market_value_in_eur"] > 0]
    v["season"] = season_of(v["date"])
    last = v.sort_values("date").groupby(["player_id", "season"]).tail(1)
    adj = adjuster(index_df)
    last["mv_adj"] = last["market_value_in_eur"] / last["season"].map(adj)
    return last[["player_id", "season", "mv_adj"]]


def prepare(scored: pd.DataFrame) -> pd.DataFrame:
    d = scored.copy()
    d["age"] = (pd.to_datetime((d["season"] + 1).astype(str) + "-01-01") - pd.to_datetime(d["date_of_birth"])).dt.days / 365.25
    d["team_ppg"] = d["team_pts"] / d["team_games"]
    return eu_role_columns(d)


def eu_role_columns(d: pd.DataFrame) -> pd.DataFrame:
    """Role-specific European goal/assist columns (explicit, so goalkeepers get no empty columns)."""
    for g in EU_ROLES:
        d[f"eu_g90_{g}"] = d["eu_z_g90"] * (d["group"] == g)
        d[f"eu_a90_{g}"] = d["eu_z_a90"] * (d["group"] == g)
    return d


def fit_market(scored: pd.DataFrame, index_df, leagues=("ES1",)):
    """leagues=("ES1",): the La Liga model used to value seasons AT the club (outcome).
    leagues=None: every league in the data, used to value players BEFORE they signed (decision)."""
    d = prepare(scored).merge(end_of_season_values(index_df), on=["player_id", "season"], how="inner")
    if leagues is not None:
        d = d[d["league"].isin(leagues)]
    d = d[(d["minutes"] > 0) & d["age"].between(16, 42) & d["group"].notna()]
    if "patched" in d:
        d = d[~d["patched"].astype(bool)]   # hand-checked seasons are valued, never used to fit
    d["log_mv"] = np.log(d["mv_adj"])
    f = FORMULA if d["league"].nunique() > 1 else FORMULA.replace("C(league) + ", "")
    model = smf.ols(f, data=d).fit()
    return model, d


def piv(model, rows: pd.DataFrame, alpha=0.2) -> pd.DataFrame:
    """Performance-implied value (inflation-adjusted euros) with an (1-alpha) prediction interval."""
    sf = model.get_prediction(rows).summary_frame(alpha=alpha)
    return pd.DataFrame({"log_piv": sf["mean"].values, "log_lo": sf["obs_ci_lower"].values,
                         "log_hi": sf["obs_ci_upper"].values}, index=rows.index)


def market_premium(min_fee=10e6) -> float:
    """Median fee / Transfermarkt value for EUR10m+ transfers into top-5-league clubs (clubs pay above valuations)."""
    from .config import TOP5
    t = load("transfers")
    c = load("clubs")[["club_id", "domestic_competition_id"]]
    t = t.merge(c, left_on="to_club_id", right_on="club_id")
    t = t[t["domestic_competition_id"].isin(TOP5) & (t["transfer_fee"] >= min_fee) & (t["market_value_in_eur"] > 0)]
    return float((t["transfer_fee"] / t["market_value_in_eur"]).median())


def market_premium_range(min_fee=10e6, q=(0.1, 0.9)):
    """10th-90th percentile of fee / Transfermarkt value for EUR10m+ transfers into top-5-league clubs."""
    from .config import TOP5
    t = load("transfers")
    c = load("clubs")[["club_id", "domestic_competition_id"]]
    t = t.merge(c, left_on="to_club_id", right_on="club_id")
    t = t[t["domestic_competition_id"].isin(TOP5) & (t["transfer_fee"] >= min_fee) & (t["market_value_in_eur"] > 0)]
    r = t["transfer_fee"] / t["market_value_in_eur"]
    return float(r.quantile(q[0])), float(r.quantile(q[1]))
