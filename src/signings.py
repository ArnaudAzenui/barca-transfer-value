"""Every senior player Barça paid a fee for from 2010/11 to 2025/26, with how and when he left."""
import numpy as np
import pandas as pd
from .config import BARCA_ID, FIRST_SEASON, CURATED, SUBPOS_TO_GROUP, POSITION_TO_GROUP
from .data import load, season_of

LAST_SEASON = 2025   # dataset ends July 2026 -> 2025/26 is the last complete season


OWN_EXACT = {418: {"real madrid", "rm castilla", "real madrid castilla", "real madrid u19"},
             13: {"atlético", "atlético madrid", "atlético de madrid", "atl. madrileño", "atlético u19", "atlético madrid b"}}
CLUB = {"id": BARCA_ID}   # set by build_signings


def _is_barca(name) -> bool:
    """True for the club itself or its reserve/youth sides (moves between them aren't transfers)."""
    if not isinstance(name, str):
        return False
    n = name.lower()
    if CLUB["id"] == BARCA_ID:
        return "barcelona" in n or "barça" in n
    return n in OWN_EXACT.get(CLUB["id"], set())


def _season(d) -> int:
    """June dates are end-of-season paperwork: a June signing belongs to the next season, a June exit ends the season."""
    d = pd.Timestamp(d)
    return d.year - (d.month < 6)


def _mv_at(vals, pid, when):
    """Latest Transfermarkt valuation on or before `when`. If that one is stale (>6 months old), use the first
    valuation within 4 months after, which usually reflects the same information the buying club had."""
    pv = vals[vals["player_id"] == pid]
    v = pv[pv["date"] <= when]
    if not v.empty and (pd.Timestamp(when) - v["date"].iloc[-1]).days <= 180:
        return float(v["market_value_in_eur"].iloc[-1])
    after = pv[(pv["date"] > when) & (pv["date"] <= pd.Timestamp(when) + pd.Timedelta(days=120))]
    if not after.empty and pd.Timestamp(when) != pd.Timestamp.max:
        return float(after["market_value_in_eur"].iloc[0])
    if not v.empty:
        return float(v["market_value_in_eur"].iloc[-1])
    return float(pv["market_value_in_eur"].iloc[0]) if not pv.empty else np.nan


def _from_transfers(t: pd.DataFrame) -> list[dict]:
    BARCA_ID = CLUB["id"]
    ins = t[(t["to_club_id"] == BARCA_ID) & (t["transfer_fee"] > 0) & (t["season"] >= FIRST_SEASON)
            & (t["season"] <= LAST_SEASON) & ~t["from_club_name"].map(_is_barca)].sort_values("transfer_date")
    rows = []
    for _, s in ins.iterrows():
        pid = s["player_id"]
        later = t[(t["player_id"] == pid) & (t["transfer_date"] > s["transfer_date"])].sort_values("transfer_date")
        nxt = later[(later["to_club_id"] == BARCA_ID) & (later["transfer_fee"] > 0) & ~later["from_club_name"].map(_is_barca)]
        if not nxt.empty:
            later = later[later["transfer_date"] < nxt["transfer_date"].iloc[0]]
        loan_income, loan_seasons, exit_row = 0.0, [], None
        outs = later[(later["from_club_id"] == BARCA_ID) & ~later["to_club_name"].map(_is_barca)]
        for _, o in outs.iterrows():
            back = later[(later["transfer_date"] > o["transfer_date"]) & (later["to_club_id"] == BARCA_ID)]
            if back.empty:
                exit_row = o
                break
            loan_income += o["transfer_fee"]
            loan_seasons += list(range(int(o["season"]), int(back["season"].iloc[0])))
        rows.append(dict(
            player_id=pid, player=s["player_name"], signing_date=s["transfer_date"], from_club=s["from_club_name"],
            fee_eur=float(s["transfer_fee"]),
            exit_date=None if exit_row is None else exit_row["transfer_date"],
            exit_to=None if exit_row is None else exit_row["to_club_name"],
            sale_fee_eur=0.0 if exit_row is None else float(exit_row["transfer_fee"]),
            loan_income_eur=loan_income, loaned_out_seasons=sorted(set(loan_seasons)),
            source="dataset", confidence="dataset"))
    return rows


def _from_curated(club_id: int = BARCA_ID) -> list[dict]:
    if club_id == BARCA_ID:
        c = pd.read_csv(CURATED / "missing_signings.csv", comment="#")
    else:
        c = pd.read_csv(CURATED / "missing_signings_rivals.csv", comment="#")
        c = c[c["club_id"] == club_id]
    rows = []
    for _, r in c.iterrows():
        loans = [int(x) for x in str(r["loaned_out_seasons"]).split(";") if x.strip() not in ("", "nan")]
        rows.append(dict(
            player_id=int(r["player_id"]), player=r["player"], signing_date=pd.Timestamp(r["signing_date"]),
            from_club=r["from_club"], fee_eur=r["fee_eur_m"] * 1e6,
            exit_date=pd.Timestamp(r["exit_date"]) if pd.notna(r["exit_date"]) else None,
            exit_to=r["exit_to"], sale_fee_eur=r["sale_fee_eur_m"] * 1e6,
            loan_income_eur=r["loan_income_eur_m"] * 1e6, loaned_out_seasons=loans,
            source="curated", confidence=r["confidence"]))
    return rows


def build_signings(club_id: int = BARCA_ID) -> pd.DataFrame:
    CLUB["id"] = club_id
    t = load("transfers").copy()
    t["season"] = season_of(t["transfer_date"])
    t["transfer_fee"] = t["transfer_fee"].fillna(0)
    # hand-curated deals missing from the dataset (fee fixes exist for Barça only)
    out = pd.DataFrame(_from_transfers(t) + _from_curated(club_id))
    out["signing_season"] = out["signing_date"].map(_season)

    ov = pd.read_csv(CURATED / "fee_overrides.csv", comment="#") if club_id == BARCA_ID else pd.DataFrame()
    for _, r in ov.iterrows():
        m = (out["player"] == r["player"]) & (out["signing_season"] == r["signing_season"]) & (out["source"] == "dataset")
        if r["action"] == "drop":
            out = out[~m]
        elif r["action"] == "fee":
            out.loc[m, "fee_eur"] = float(r["fee_eur_m"]) * 1e6

    players = load("players").set_index("player_id")
    vals = load("player_valuations").sort_values("date")
    app = load("appearances")
    app = app[app["competition_id"] == "ES1"].copy()
    gs = load("games")[["game_id", "season"]]
    app = app.merge(gs, on="game_id", how="left")   # official season label (2019/20 ran into July 2020)
    barca_apps = app[app["player_club_id"] == club_id]
    other_apps = load("appearances")[lambda d: d["player_club_id"] != club_id].copy()
    other_apps["season"] = season_of(other_apps["date"])

    rec = []
    for _, s in out.sort_values("signing_date").iterrows():
        pid = s["player_id"]
        p = players.loc[pid] if pid in players.index else pd.Series(dtype=object)
        group = SUBPOS_TO_GROUP.get(p.get("sub_position"), POSITION_TO_GROUP.get(p.get("position"), "MID"))
        dob = p.get("date_of_birth")
        ongoing = s["exit_date"] is None or pd.isna(s["exit_date"])
        end = pd.Timestamp("2026-07-01") if ongoing else pd.Timestamp(s["exit_date"])
        exit_season = LAST_SEASON + 1 if ongoing else _season(end)

        seasons = set(range(int(s["signing_season"]), exit_season)) - set(s["loaned_out_seasons"])
        # seasons where he actually played league games for Barça inside the stint (catches late exits)
        played = barca_apps[(barca_apps["player_id"] == pid) & (barca_apps["date"] >= s["signing_date"])
                            & (barca_apps["date"] <= end)]["season"].dropna().astype(int)
        seasons |= set(played)
        # seasons spent entirely at another club (loan-backs / unrecorded loans) are not Barça seasons
        elsewhere = set(other_apps[(other_apps["player_id"] == pid)]["season"].dropna().astype(int))
        seasons = {y for y in seasons if y in set(played) or y not in elsewhere}
        seasons = sorted(y for y in seasons if y <= LAST_SEASON)

        rec.append({**s.to_dict(),
                    "group": group, "sub_position": p.get("sub_position"),
                    "age_at_signing": (s["signing_date"] - dob).days / 365.25 if pd.notna(dob) else np.nan,
                    "mv_at_signing_eur": _mv_at(vals, pid, s["signing_date"]),
                    "mv_latest_eur": _mv_at(vals, pid, end),
                    "ongoing": ongoing, "exit_season": None if ongoing else exit_season,
                    "barca_seasons": seasons})
    return pd.DataFrame(rec).reset_index(drop=True)
