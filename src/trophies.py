"""Trophies: who won what, and how much each player contributed to winning it.

Winners are derived from the match data:
  * La Liga: most points (tie-break: head-to-head points, then head-to-head goal difference)
  * Cup competitions: the final (scores include penalty shoot-outs); two-legged finals on aggregate, then away goals

Trophy points for a player-season = sum over competitions his club WON of
    TROPHY_WEIGHTS[competition] x involvement
    involvement = 0.6 x (his share of the team's minutes in that competition)
                + 0.4 x (his goals + assists / team goals in that competition)      [both capped at 1]
So a regular who scored and assisted in a Champions League win earns close to the full 3 points; a squad player
who played two group games earns a fraction; someone who didn't play in the competition earns nothing.
"""
import numpy as np
import pandas as pd
from .config import TROPHY_WEIGHTS, TROPHY_NAMES, TROPHY_TYPE_WEIGHTS, PLAYOFF_LEAGUES
from .data import load


def weights_and_names():
    """Weight and display name for every competition: explicit ones first, then by competition type."""
    c = load("competitions")
    w, n = {}, {}
    for _, r in c.iterrows():
        cid = r["competition_id"]
        if cid in TROPHY_WEIGHTS:
            w[cid], n[cid] = TROPHY_WEIGHTS[cid], TROPHY_NAMES[cid]
        elif r["type"] in TROPHY_TYPE_WEIGHTS and not (r["type"] == "domestic_league" and cid in PLAYOFF_LEAGUES):
            w[cid] = TROPHY_TYPE_WEIGHTS[r["type"]]
        elif r["sub_type"] in TROPHY_TYPE_WEIGHTS:
            w[cid] = TROPHY_TYPE_WEIGHTS[r["sub_type"]]
        if cid in w and cid not in n:
            n[cid] = str(r["name"]).replace("-", " ").title()
    for cid, wt in TROPHY_WEIGHTS.items():       # e.g. the Club World Cup isn't in the competitions table
        w.setdefault(cid, wt); n.setdefault(cid, TROPHY_NAMES[cid])
    return w, n


W, NAMES = None, None


def _wn():
    global W, NAMES
    if W is None:
        W, NAMES = weights_and_names()
    return W, NAMES


def _league_ids():
    c = load("competitions")
    return [x for x in c[c["type"] == "domestic_league"]["competition_id"] if x not in PLAYOFF_LEAGUES]


def league_champions() -> pd.DataFrame:
    g = load("games")
    g = g[g["competition_id"].isin(_league_ids())].dropna(subset=["home_club_goals", "away_club_goals"])
    rows = []
    for (season, comp), s in g.groupby(["season", "competition_id"]):
        n_clubs = len(set(s["home_club_id"]) | set(s["away_club_id"]))
        if len(s) < 0.95 * n_clubs * (n_clubs - 1):      # incomplete season: no champion yet
            continue
        pts = {}
        for _, x in s.iterrows():
            h, a, hg, ag = x["home_club_id"], x["away_club_id"], x["home_club_goals"], x["away_club_goals"]
            pts[h] = pts.get(h, 0) + (3 if hg > ag else 1 if hg == ag else 0)
            pts[a] = pts.get(a, 0) + (3 if ag > hg else 1 if hg == ag else 0)
        top = max(pts.values())
        leaders = [c for c, p in pts.items() if p == top]
        if len(leaders) > 1:   # head-to-head
            h2h = {c: 0 for c in leaders}
            gd = {c: 0 for c in leaders}
            m = s[s["home_club_id"].isin(leaders) & s["away_club_id"].isin(leaders)]
            for _, x in m.iterrows():
                h, a, hg, ag = x["home_club_id"], x["away_club_id"], x["home_club_goals"], x["away_club_goals"]
                h2h[h] += 3 if hg > ag else 1 if hg == ag else 0
                h2h[a] += 3 if ag > hg else 1 if hg == ag else 0
                gd[h] += hg - ag; gd[a] += ag - hg
            leaders = sorted(leaders, key=lambda c: (h2h[c], gd[c]), reverse=True)
        rows.append(dict(season=int(season), competition_id=comp, club_id=int(leaders[0])))
    return pd.DataFrame(rows)


def cup_winners() -> pd.DataFrame:
    w, _ = _wn()
    cups = [c for c in w if c not in set(_league_ids())]
    g = load("games")
    g = g[g["competition_id"].isin(cups)].dropna(subset=["home_club_goals", "away_club_goals"])
    rnd = g["round"].fillna("").str.lower()
    finals = g[rnd.str.startswith("final")]
    rows = []
    for (season, comp), f in finals.groupby(["season", "competition_id"]):
        if len(f) == 1:
            x = f.iloc[0]
            if x["home_club_goals"] == x["away_club_goals"]:
                continue   # shouldn't happen (shoot-outs are in the score)
            w = x["home_club_id"] if x["home_club_goals"] > x["away_club_goals"] else x["away_club_id"]
        else:              # two legs: aggregate, then away goals
            clubs = list(set(f["home_club_id"]) | set(f["away_club_id"]))
            tot = {c: 0 for c in clubs}; away = {c: 0 for c in clubs}
            for _, x in f.iterrows():
                tot[x["home_club_id"]] += x["home_club_goals"]; tot[x["away_club_id"]] += x["away_club_goals"]
                away[x["away_club_id"]] += x["away_club_goals"]
            w = sorted(clubs, key=lambda c: (tot[c], away[c]), reverse=True)[0]
        rows.append(dict(season=int(season), competition_id=comp, club_id=int(w)))
    return pd.DataFrame(rows)


def winners() -> pd.DataFrame:
    return pd.concat([league_champions(), cup_winners()], ignore_index=True)


def trophy_points() -> pd.DataFrame:
    """One row per (player, club, season) with trophy points and the list of trophies he contributed to."""
    wts, names = _wn()
    w = winners()
    g = load("games")[["game_id", "season", "competition_id", "home_club_id", "away_club_id", "home_club_goals", "away_club_goals"]]
    g = g[g["competition_id"].isin(wts)]
    team = pd.concat([
        g.rename(columns={"home_club_id": "club_id", "home_club_goals": "gf"})[["game_id", "season", "competition_id", "club_id", "gf"]],
        g.rename(columns={"away_club_id": "club_id", "away_club_goals": "gf"})[["game_id", "season", "competition_id", "club_id", "gf"]]])
    team = team.merge(w, on=["season", "competition_id", "club_id"])        # only games of the eventual winner
    tstats = team.groupby(["club_id", "season", "competition_id"]).agg(team_games=("game_id", "nunique"),
                                                                        team_goals=("gf", "sum")).reset_index()
    a = load("appearances")[["game_id", "player_id", "player_club_id", "minutes_played", "goals", "assists"]]
    a = a.merge(team[["game_id", "club_id", "season", "competition_id"]], left_on=["game_id", "player_club_id"],
                right_on=["game_id", "club_id"])
    p = a.groupby(["player_id", "club_id", "season", "competition_id"]).agg(
        minutes=("minutes_played", "sum"), ga=("goals", "sum"), ga2=("assists", "sum")).reset_index()
    p["ga"] = p["ga"] + p["ga2"]
    p = p.merge(tstats, on=["club_id", "season", "competition_id"])
    p["min_share"] = (p["minutes"] / (p["team_games"] * 90)).clip(upper=1)
    p["ga_share"] = (p["ga"] / p["team_goals"].replace(0, np.nan)).fillna(0).clip(upper=1)
    p["involvement"] = 0.6 * p["min_share"] + 0.4 * p["ga_share"]
    p["points"] = p["competition_id"].map(wts) * p["involvement"]
    p["label"] = p["competition_id"].map(names) + " (" + (p["involvement"] * 100).round().astype(int).astype(str) + "%)"
    out = p.groupby(["player_id", "club_id", "season"]).agg(
        trophy_pts=("points", "sum"),
        trophies=("label", lambda s: ", ".join(s))).reset_index()
    return out


def add_trophies(scored: pd.DataFrame) -> pd.DataFrame:
    t = trophy_points()
    d = scored.drop(columns=[c for c in ("trophy_pts", "trophies") if c in scored.columns])
    d = d.merge(t, on=["player_id", "club_id", "season"], how="left")
    d["trophy_pts"] = d["trophy_pts"].fillna(0.0)
    d["trophies"] = d["trophies"].fillna("")
    from .patches import apply_trophies
    return apply_trophies(d)   # La Liga / Champions League titles in 2010/11-2011/12
