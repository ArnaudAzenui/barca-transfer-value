"""Hand-checked season stats that fill gaps in the transfermarkt dataset (see data/curated/*statmuse*.csv).

Three kinds of patch, all from StatMuse (FBref for assists where StatMuse has none):
  * stat_corrections.csv      – a few goal counts in the dataset that disagree with StatMuse.
  * season_stats_statmuse.csv – player-seasons missing from the dataset:
        - 2010/11 and 2011/12 (the dataset's appearances start in 2012/13). These are "reconstructed": scored
          against the 2012/13 La Liga / Champions League peers in the same role, because no league-wide data
          exists for those seasons.
        - three 2020/21 seasons (Jordi Alba, ter Stegen, Frenkie de Jong) the dataset lost entirely.
  * team_seasons_statmuse.csv – Barça's team totals for 2010/11 and 2011/12 (points, goals, trophies).

What a patched season can't have: per-game line-ups, so points-per-game with/without the player is set to neutral
(0) and goals conceded while he played is the team's average. Missing assists are estimated from the player's own
assist rate in his other Barça seasons. Patched rows never enter the market-model fit or the peer averages.
"""
import numpy as np
import pandas as pd
from .config import CURATED

STATS, TEAMS, CORR = "season_stats_statmuse.csv", "team_seasons_statmuse.csv", "stat_corrections.csv"
NORM_SEASON = 2012   # first season with league-wide data: the yardstick for reconstructed seasons


def _read(name):
    f = CURATED / name
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def _rate(ps, pid, club, num, den):
    """Player's own per-minute rate in his other (dataset) seasons at the club."""
    x = ps[(ps["player_id"] == pid) & (ps["club_id"] == club)]
    return x[num].sum() / x[den].sum() if len(x) and x[den].sum() > 0 else 0.0


def apply_league(ps: pd.DataFrame, team: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """Called inside performance.player_seasons, before per-90 metrics are computed."""
    ps = ps.copy()
    ps["patched"], ps["reconstructed"], ps["norm_season"] = False, False, ps["season"]
    ps["patch_note"] = ""
    c = _read(CORR)
    for _, r in c.iterrows():
        m = (ps["player_id"] == r["player_id"]) & (ps["club_id"] == r["club_id"]) & (ps["season"] == r["season"]) & (ps["league"] == r["league"])
        ps.loc[m, r["field"]] = r["value"]
        ps.loc[m, "patch_note"] = f"{r['field']} corrected {r['dataset_value']}→{r['value']} ({r['source']})"
    s, t = _read(STATS), _read(TEAMS)
    rows = []
    for _, r in s.iterrows():
        key = (r["player_id"], r["club_id"], r["season"])
        if ((ps["player_id"] == key[0]) & (ps["club_id"] == key[1]) & (ps["season"] == key[2]) & (ps["league"] == "ES1")).any():
            continue
        tt = t[(t["club_id"] == r["club_id"]) & (t["season"] == r["season"])]
        if len(tt):
            tt = tt.iloc[0]
            games, pts, ga, coverage = tt["league_games"], tt["league_pts"], tt["league_ga"], 1.0
        else:
            tm = team[(team["club_id"] == r["club_id"]) & (team["season"] == r["season"]) & (team["league"] == "ES1")].iloc[0]
            games, pts, ga = tm["team_games"], tm["team_pts"], tm["team_ga"]
            cv = cov[(cov["club_id"] == r["club_id"]) & (cov["season"] == r["season"]) & (cov["league"] == "ES1")]
            coverage = float(cv["coverage"].iloc[0]) if len(cv) else 1.0
        assists, note = r["league_assists"], ""
        if pd.isna(assists):
            assists = round(_rate(ps, r["player_id"], r["club_id"], "assists", "minutes") * r["league_min"], 1)
            note = "league assists estimated from his other Barça seasons"
        g60 = r["league_starts"]
        rows.append(dict(player_id=r["player_id"], club_id=r["club_id"], season=r["season"], league="ES1",
                         minutes=r["league_min"], goals=r["league_goals"], assists=assists, apps=r["league_apps"],
                         g60=g60, pts_on=g60 * pts / games, ga_on=g60 * ga / games,
                         team_games=games, team_pts=pts, team_ga=ga, coverage=coverage, team_minutes=np.nan,
                         patched=True, reconstructed=r["season"] < NORM_SEASON,
                         norm_season=max(int(r["season"]), NORM_SEASON), patch_note=note))
    if rows:
        ps = pd.concat([ps, pd.DataFrame(rows)], ignore_index=True)
    return ps


def apply_europe(d: pd.DataFrame, norms: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Champions League output for patched seasons, z-scored against CL peers (same role, norm season)."""
    s, t = _read(STATS), _read(TEAMS)
    if s.empty or "patched" not in d:
        return d
    from .europe import SHRINK, EU_METRICS, cl_team_games
    d = d.copy()
    tg = cl_team_games()
    for _, r in s.iterrows():
        m = (d["player_id"] == r["player_id"]) & (d["club_id"] == r["club_id"]) & (d["season"] == r["season"]) & d["patched"]
        if not m.any():
            continue
        i = d.index[m][0]
        tt = t[(t["club_id"] == r["club_id"]) & (t["season"] == r["season"])]
        games = tt["cl_games"].iloc[0] if len(tt) else tg.get((r["club_id"], r["season"]), 8)
        grp, ns = d.at[i, "group"], d.at[i, "norm_season"]
        mins = r["cl_min"]
        a = r["cl_assists"]
        note = d.at[i, "patch_note"]
        if pd.isna(a):
            a = round(_rate(d[~d["patched"]], r["player_id"], r["club_id"], "eu_assists", "eu_minutes") * mins, 1)
            note = "; ".join(x for x in [note, "Champions League assists estimated from his other Barça seasons"] if x)
        vals = {"g90": r["cl_goals"] / max(mins, 1) * 90, "a90": a / max(mins, 1) * 90, "mins": mins / (games * 90)}
        for k in EU_METRICS:
            mu, sd = norms.get((ns, grp, k), (np.nan, np.nan))
            z = float(np.nan_to_num(np.clip((vals[k] - mu) / sd, -3, 3))) if sd and sd > 0 else 0.0
            if k != "mins":
                z *= mins / (mins + SHRINK)
            d.at[i, f"eu_z_{k}"] = z
        if grp == "GK":
            d.at[i, "eu_z_g90"] = d.at[i, "eu_z_a90"] = 0.0
        d.loc[i, ["eu_minutes", "eu_goals", "eu_assists", "eu_apps", "cl_minutes"]] = [mins, r["cl_goals"], a, r["cl_apps"], mins]
        d.at[i, "eu_comps"], d.at[i, "team_europe"], d.at[i, "in_cl"], d.at[i, "in_el"] = "CL", "CL", 1, 0
        w = weights.get(grp)
        if isinstance(w, dict):
            ws = {k: w[k] for k in EU_METRICS}
            d.at[i, "eu_score"] = sum(ws[k] * d.at[i, f"eu_z_{k}"] for k in EU_METRICS) / (sum(ws.values()) or 1)
        d.at[i, "patch_note"] = note
    return d


def apply_trophies(d: pd.DataFrame) -> pd.DataFrame:
    """La Liga and Champions League titles in reconstructed seasons, with the same involvement formula as
    trophies.py (0.6 x share of the team's minutes + 0.4 x share of the team's goals he scored or assisted)."""
    t = _read(TEAMS)
    if t.empty or "reconstructed" not in d:
        return d
    from .config import TROPHY_WEIGHTS, TROPHY_NAMES
    d = d.copy()
    for i in d.index[d["reconstructed"]]:
        r = d.loc[i]
        tt = t[(t["club_id"] == r["club_id"]) & (t["season"] == r["season"])]
        if tt.empty:
            continue
        tt = tt.iloc[0]
        pts, labels = 0.0, []
        for comp, won, games, goals, mins, ga in [
                ("ES1", tt["won_league"], tt["league_games"], tt["league_gf"], r["minutes"], r["goals"] + r["assists"]),
                ("CL", tt["won_cl"], tt["cl_games"], tt["cl_goals"], r["eu_minutes"], r["eu_goals"] + r["eu_assists"])]:
            if not won:
                continue
            inv = 0.6 * min(1, mins / (games * 90)) + 0.4 * min(1, ga / goals)
            pts += TROPHY_WEIGHTS[comp] * inv
            labels.append(f"{TROPHY_NAMES[comp]} ({round(inv * 100)}%)")
        d.at[i, "trophy_pts"], d.at[i, "trophies"] = pts, ", ".join(labels)
    return d
