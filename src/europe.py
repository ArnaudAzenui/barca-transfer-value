"""European matches (Champions League + Europa League) for every player-season.

For each competition and season, a player's goals/90, assists/90 and share of his team's European minutes are
z-scored against every player in that competition *in the same role group* (peers need 270+ minutes, i.e. three
full games). Per-90 z-scores shrink toward 0 below ~3 games. If a player appeared in both competitions in a season
(e.g. Barça in 2021/22 dropped from the CL into the EL), the two are combined, weighted by minutes.
Champions League output counts fully; Europa League output is discounted (EL_WEIGHT) because the opposition is weaker.
"""
import numpy as np
import pandas as pd
from .config import SUBPOS_TO_GROUP, POSITION_TO_GROUP, ROLE_WEIGHTS
from .data import load

COMPS = ["CL", "EL"]
EL_WEIGHT = 0.6
MIN_PEER = 270
SHRINK = 270
EU_METRICS = ["g90", "a90", "mins"]


def europe_seasons() -> pd.DataFrame:
    g = load("games")[["game_id", "season", "competition_id", "home_club_id", "away_club_id"]]
    g = g[g["competition_id"].isin(COMPS)]
    a = load("appearances")
    a = a[a["competition_id"].isin(COMPS)].merge(g[["game_id", "season"]], on="game_id")
    team_games = pd.concat([g[["game_id", "season", "competition_id", "home_club_id"]].rename(columns={"home_club_id": "club_id"}),
                            g[["game_id", "season", "competition_id", "away_club_id"]].rename(columns={"away_club_id": "club_id"})])
    tg = team_games.groupby(["club_id", "season", "competition_id"])["game_id"].nunique().rename("eu_team_games").reset_index()
    ps = a.groupby(["player_id", "player_club_id", "season", "competition_id"]).agg(
        eu_minutes=("minutes_played", "sum"), eu_goals=("goals", "sum"), eu_assists=("assists", "sum"),
        eu_apps=("game_id", "nunique")).reset_index().rename(columns={"player_club_id": "club_id"})
    ps = ps.merge(tg, on=["club_id", "season", "competition_id"], how="left")
    pl = load("players")[["player_id", "position", "sub_position"]]
    ps = ps.merge(pl, on="player_id", how="left")
    ps["group"] = ps["sub_position"].map(SUBPOS_TO_GROUP).fillna(ps["position"].map(POSITION_TO_GROUP))
    m = ps["eu_minutes"].clip(lower=1)
    ps["g90"] = ps["eu_goals"] / m * 90
    ps["a90"] = ps["eu_assists"] / m * 90
    ps["mins"] = ps["eu_minutes"] / (ps["eu_team_games"] * 90)

    peers = ps[ps["eu_minutes"] >= MIN_PEER]
    st = peers.groupby(["competition_id", "season", "group"])[EU_METRICS].agg(["mean", "std"])
    key = ps.set_index(["competition_id", "season", "group"]).index
    for mtr in EU_METRICS:
        mu = np.asarray(key.map(st[(mtr, "mean")]), float)
        sd = np.asarray(key.map(st[(mtr, "std")]), float)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.nan_to_num(np.clip((ps[mtr].values - mu) / sd, -3, 3))
        if mtr != "mins":
            z = z * ps["eu_minutes"] / (ps["eu_minutes"] + SHRINK)
        ps[f"eu_z_{mtr}"] = z

    # combine CL + EL per player-club-season: minutes-weighted z, EL discounted
    ps["w"] = ps["eu_minutes"].clip(lower=1) * np.where(ps["competition_id"] == "EL", EL_WEIGHT, 1.0)
    agg = {f"eu_z_{k}": (lambda s, k=k: np.average(s, weights=ps.loc[s.index, "w"])) for k in EU_METRICS}
    out = ps.groupby(["player_id", "club_id", "season"]).agg(
        eu_minutes=("eu_minutes", "sum"), eu_goals=("eu_goals", "sum"), eu_assists=("eu_assists", "sum"),
        eu_apps=("eu_apps", "sum"), eu_comps=("competition_id", lambda s: "+".join(sorted(set(s)))),
        cl_minutes=("eu_minutes", lambda s: s[ps.loc[s.index, "competition_id"] == "CL"].sum()),
        **{f"eu_z_{k}": (f"eu_z_{k}", "mean") for k in EU_METRICS}).reset_index()
    # minutes-weighted combination (overwrite the simple means)
    for k in EU_METRICS:
        ps[f"_wz_{k}"] = ps[f"eu_z_{k}"] * ps["w"]
    wsum = ps.groupby(["player_id", "club_id", "season"])[["w"] + [f"_wz_{k}" for k in EU_METRICS]].sum().reset_index()
    for k in EU_METRICS:
        wsum[f"eu_z_{k}"] = wsum[f"_wz_{k}"] / wsum["w"]
    out = out.drop(columns=[f"eu_z_{k}" for k in EU_METRICS]).merge(
        wsum[["player_id", "club_id", "season"] + [f"eu_z_{k}" for k in EU_METRICS]], on=["player_id", "club_id", "season"])
    return out


def team_in_europe() -> pd.DataFrame:
    """(club, season) pairs that played CL or EL group/knockout games, with the competition level."""
    g = load("games")[["season", "competition_id", "home_club_id", "away_club_id"]]
    g = g[g["competition_id"].isin(COMPS)]
    t = pd.concat([g.rename(columns={"home_club_id": "club_id"}), g.rename(columns={"away_club_id": "club_id"})])
    t = t.groupby(["club_id", "season"])["competition_id"].agg(lambda s: "CL" if "CL" in set(s) else "EL").reset_index()
    return t.rename(columns={"competition_id": "team_europe"})


def add_europe(scored: pd.DataFrame) -> pd.DataFrame:
    """Attach European metrics to La Liga player-seasons. Players whose club wasn't in Europe get zeros
    (neutral), players at a European club who didn't play get their (negative) minutes z-score."""
    eu = europe_seasons()
    te = team_in_europe()
    d = scored.merge(te, on=["club_id", "season"], how="left")
    d = d.merge(eu, on=["player_id", "club_id", "season"], how="left")
    d["team_europe"] = d["team_europe"].fillna("none")
    for c in ["eu_minutes", "eu_goals", "eu_assists", "eu_apps", "cl_minutes"]:
        d[c] = d[c].fillna(0)
    # at a European club but never used: same floor as a near-zero-minutes player
    unused = (d["team_europe"] != "none") & d["eu_z_mins"].isna()
    d.loc[unused, "eu_z_mins"] = -3.0
    for k in EU_METRICS:
        d[f"eu_z_{k}"] = d[f"eu_z_{k}"].fillna(0.0)
    # goalkeepers: a single freak goal/assist would dominate the fit, so only their European minutes count
    gk = d["group"] == "GK"
    d.loc[gk, ["eu_z_g90", "eu_z_a90"]] = 0.0
    d["in_cl"] = (d["team_europe"] == "CL").astype(int)
    d["in_el"] = (d["team_europe"] == "EL").astype(int)
    # European role score (same role weights as the league score, renormalised over the metrics we have)
    def eu_score(r):
        w = ROLE_WEIGHTS.get(r["group"])
        if not isinstance(w, dict) or r["team_europe"] == "none":
            return np.nan
        ws = {k: w[k] for k in EU_METRICS}
        tot = sum(ws.values()) or 1
        return sum(ws[k] * r[f"eu_z_{k}"] for k in EU_METRICS) / tot
    d["eu_score"] = d.apply(eu_score, axis=1)
    return d
