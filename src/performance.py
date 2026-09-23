"""Role-aware league performance scores for every player-season, benchmarked against positional peers.

Every first-tier domestic league in the dataset is scored (La Liga, the other top-5 leagues, Portugal, the
Netherlands, ...), each against its own players: a z-score always means "compared with the same role in the same
league and season". League strength is handled later by league fixed effects in the market model."""
import numpy as np
import pandas as pd
from .config import (LALIGA, BARCA_ID, SUBPOS_TO_GROUP, POSITION_TO_GROUP, ROLE_WEIGHTS,
                     MIN_PEER_MINUTES, SHRINK_MINUTES)
from .data import load, season_of

METRICS = ["g90", "a90", "mins", "onoff", "cs"]
PER_GAME_METRICS = ["g90", "a90", "onoff", "cs"]   # noisy at low minutes -> shrink


def league_ids() -> list:
    c = load("competitions")
    return c[c["type"] == "domestic_league"]["competition_id"].tolist()


def team_games() -> pd.DataFrame:
    """One row per (club, league game) with points and goals against."""
    g = load("games")
    g = g[g["competition_id"].isin(league_ids())].copy()
    # use the official season label: 2019/20 finished in July 2020 (COVID), so dates alone would misfile it
    g["season"] = g["season"].astype(int)
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        x = g[["game_id", "season", "competition_id", f"{side}_club_id", f"{side}_club_goals", f"{opp}_club_goals"]].copy()
        x.columns = ["game_id", "season", "league", "club_id", "gf", "ga"]
        rows.append(x)
    tg = pd.concat(rows).dropna(subset=["gf", "ga"])
    tg["pts"] = np.select([tg.gf > tg.ga, tg.gf == tg.ga], [3, 1], 0)
    return tg


def player_seasons() -> pd.DataFrame:
    a = load("appearances")
    a = a[a["competition_id"].isin(league_ids())].copy()
    tg_all = team_games()
    # A few club-seasons have almost no line-up data (e.g. Atlético 2014/15). Team totals use the games we have
    # line-ups for, and `coverage` (recorded minutes / a full season of 38 x 11 x 90) flags the gaps; seasons below
    # 50% coverage are treated like the pre-2012 seasons (credited at the player's average).
    covered = a[["game_id", "player_club_id"]].drop_duplicates().rename(columns={"player_club_id": "club_id"})
    tg = tg_all.merge(covered, on=["game_id", "club_id"])
    # coverage = recorded line-up minutes / a full season of them (games x 11 players x 90 minutes)
    n_clubs = tg_all.groupby(["league", "season"])["club_id"].nunique()
    expected = (2 * (n_clubs - 1)).rename("expected_games").reset_index()
    a = a.merge(tg, left_on=["game_id", "player_club_id"], right_on=["game_id", "club_id"], how="inner",
                suffixes=("", "_g"))
    rec = a.groupby(["player_club_id", "season", "league"])["minutes_played"].sum().rename("team_minutes").reset_index()
    rec = rec.rename(columns={"player_club_id": "club_id"})
    cov = rec.merge(expected, on=["league", "season"])
    cov["coverage"] = (cov["team_minutes"] / (cov["expected_games"] * 990)).clip(upper=1)
    cov = cov[["club_id", "season", "league", "coverage", "team_minutes"]]
    a["played60"] = a["minutes_played"] >= 60
    a["pts60"] = a["pts"] * a["played60"]
    a["ga60"] = a["ga"] * a["played60"]

    team = tg.groupby(["club_id", "season", "league"]).agg(team_games=("game_id", "nunique"),
                                                 team_pts=("pts", "sum"), team_ga=("ga", "sum")).reset_index()
    ps = a.groupby(["player_id", "player_club_id", "season", "league"]).agg(
        minutes=("minutes_played", "sum"), goals=("goals", "sum"), assists=("assists", "sum"),
        apps=("game_id", "nunique"), g60=("played60", "sum"),
        pts_on=("pts60", "sum"), ga_on=("ga60", "sum"),
    ).reset_index().rename(columns={"player_club_id": "club_id"})
    ps = ps.merge(team, on=["club_id", "season", "league"], how="left")
    ps = ps.merge(cov, on=["club_id", "season", "league"], how="left")

    m = ps["minutes"].clip(lower=1)
    ps["g90"] = ps["goals"] / m * 90
    ps["a90"] = ps["assists"] / m * 90
    ps["mins"] = ps["minutes"] / (ps["team_games"] * 90)
    off = ps["team_games"] - ps["g60"]
    ppg_on = ps["pts_on"] / ps["g60"].replace(0, np.nan)
    ppg_off = (ps["team_pts"] - ps["pts_on"]) / off.replace(0, np.nan)
    # need a few games each side for on/off to mean anything; otherwise neutral
    ps["onoff"] = np.where((ps["g60"] >= 5) & (off >= 3), ppg_on - ppg_off, np.nan)
    ps["cs"] = -(ps["ga_on"] / ps["g60"].replace(0, np.nan))     # fewer conceded = higher

    pl = load("players")[["player_id", "name", "position", "sub_position", "date_of_birth"]]
    ps = ps.merge(pl, on="player_id", how="left")
    ps["group"] = ps["sub_position"].map(SUBPOS_TO_GROUP).fillna(ps["position"].map(POSITION_TO_GROUP))
    return ps


def score(ps: pd.DataFrame) -> pd.DataFrame:
    """z-score each metric within (league, season, role group) using peers with enough minutes; combine by role weights."""
    peers = ps[ps["minutes"] >= MIN_PEER_MINUTES]
    stats = peers.groupby(["league", "season", "group"])[METRICS].agg(["mean", "std"])
    out = ps.copy()
    key = out.set_index(["league", "season", "group"]).index
    for mtr in METRICS:
        mu = key.map(stats[(mtr, "mean")])
        sd = key.map(stats[(mtr, "std")])
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (out[mtr].values - np.asarray(mu, float)) / np.asarray(sd, float)
        z = np.nan_to_num(np.clip(z, -3, 3))
        if mtr in PER_GAME_METRICS:
            z = z * out["minutes"] / (out["minutes"] + SHRINK_MINUTES)
        out[f"z_{mtr}"] = z
    W = pd.DataFrame(list(out["group"].map(lambda g: ROLE_WEIGHTS.get(g, {k: np.nan for k in METRICS}))),
                     index=out.index)
    out["score"] = sum(W[k] * out[f"z_{k}"] for k in METRICS)
    out["peer_mins_mean"] = key.map(stats[("mins", "mean")])
    out["peer_mins_sd"] = key.map(stats[("mins", "std")])
    return out


def zero_season_score(group: str, season: int, scored: pd.DataFrame) -> float:
    """Score for a season on Barça's books with no league minutes at all (injury / frozen out)."""
    row = scored[(scored["group"] == group) & (scored["season"] == season) & (scored["league"] == LALIGA)].head(1)
    if row.empty:
        return np.nan
    z = np.clip((0 - row["peer_mins_mean"].iloc[0]) / row["peer_mins_sd"].iloc[0], -3, 3)
    return ROLE_WEIGHTS[group]["mins"] * z
