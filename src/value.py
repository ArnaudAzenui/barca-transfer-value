"""Return side per signing, and the value-for-money model."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from .config import BARCA_ID
from .performance import zero_season_score

FIRST_COVERED = 2012   # appearance data starts in 2012/13


def add_performance(s: pd.DataFrame, scored: pd.DataFrame, club_id: int = BARCA_ID) -> pd.DataFrame:
    own = scored[(scored["club_id"] == club_id) & (scored["league"] == "ES1")]
    b = own.set_index(["player_id", "season"])
    cov = own.groupby("season")["coverage"].max()
    rows = []
    for _, r in s.iterrows():
        per, mins, goals, assists = [], 0, 0, 0
        eu_per, eu_mins, eu_goals, eu_assists, cl_mins = [], 0, 0, 0, 0
        tpts, tlist = 0.0, []
        # a season counts as covered when line-up data exists for at least half of the club's league games
        # ... or when a hand-checked (StatMuse) season exists for him, e.g. 2010/11-2011/12
        covered = [y for y in r["barca_seasons"] if (y >= FIRST_COVERED and cov.get(y, 0) >= 0.5)
                   or (r["player_id"], y) in b.index]
        for y in covered:
            if (r["player_id"], y) in b.index:
                x = b.loc[(r["player_id"], y)]
                per.append(float(x["score"])); mins += x["minutes"]; goals += x["goals"]; assists += x["assists"]
                if "eu_minutes" in x:
                    eu_mins += x["eu_minutes"]; eu_goals += x["eu_goals"]; eu_assists += x["eu_assists"]
                    cl_mins += x["cl_minutes"]
                    eu_per.append(float(x["eu_score"]) if pd.notna(x["eu_score"]) else np.nan)
                if "trophy_pts" in x:
                    tpts += float(x["trophy_pts"])
                    if x["trophies"]:
                        tlist.append(f"{y}/{str(y + 1)[2:]}: {x['trophies']}")
            else:
                per.append(zero_season_score(r["group"], y, scored))
        n_all, n_cov = len(r["barca_seasons"]), len(covered)
        pas_cov = sum(max(0.0, 1 + p) for p in per if pd.notna(p))
        rows.append({
            "seasons": n_all, "seasons_covered": n_cov,
            "avg_score": np.nanmean(per) if per else np.nan,
            "season_scores": per,
            # Performance-Adjusted Seasons: 1.0 ~ one season as an average La Liga regular in his role.
            # Seasons before 2012/13 (no data) are credited at the player's covered-season average.
            "pas": pas_cov * (n_all / n_cov) if n_cov else 0.0,
            "liga_minutes": mins, "liga_goals": goals, "liga_assists": assists,
            "eu_minutes": eu_mins, "eu_goals": eu_goals, "eu_assists": eu_assists, "cl_minutes": cl_mins,
            "eu_season_scores": eu_per,
            "trophy_pts": tpts, "trophies": " | ".join(tlist),
            "avg_eu_score": np.nanmean(eu_per) if eu_per and not all(pd.isna(eu_per)) else np.nan,
        })
    return pd.concat([s.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def pos3(g):
    return {"GK": "DEF", "CB": "DEF", "FB": "DEF", "MID": "MID", "AM": "ATT", "FW": "ATT"}[g]


# ---------------------------------------------------------------------------------------------
# Market-based test: fee vs Performance-Implied Value (PIV) from the league-wide market model
# ---------------------------------------------------------------------------------------------
Z80 = 1.2816
PRIME_AGE = 27


def add_piv(s: pd.DataFrame, scored: pd.DataFrame, market_model, premium: float = 1.0, club_id: int = BARCA_ID) -> pd.DataFrame:
    """premium: typical fee / market-value ratio, so 'fair fee' = premium x PIV."""
    from .market import prepare
    from .performance import zero_season_score
    sigma = np.sqrt(market_model.scale)
    p = prepare(scored)
    own = p[(p["club_id"] == club_id) & (p["league"] == "ES1")]
    b = own.set_index(["player_id", "season"])
    cov = own.groupby("season")["coverage"].max()
    team = own.groupby("season")["team_ppg"].first()
    europe_status = own.groupby("season")["team_europe"].first().to_dict()
    dob = load_dob()
    out = []
    for _, r in s.iterrows():
        rows = []
        for y in [y for y in r["barca_seasons"] if (y >= FIRST_COVERED and cov.get(y, 0) >= 0.5)
                  or (r["player_id"], y) in b.index]:
            if (r["player_id"], y) in b.index:
                rows.append(b.loc[(r["player_id"], y)].to_dict() | {"season": y})
            else:   # on the books but no league minutes
                grp = scored[(scored["group"] == r["group"]) & (scored["season"] == y)].head(1)
                z_mins = float(np.clip((0 - grp["peer_mins_mean"].iloc[0]) / grp["peer_mins_sd"].iloc[0], -3, 3))
                d0 = dob.get(r["player_id"])
                age = (pd.Timestamp(f"{y + 1}-01-01") - d0).days / 365.25 if pd.notna(d0) else r["age_at_signing"] + (y - r["signing_season"])
                te = europe_status.get(y, "none")
                rows.append(dict(season=y, league="ES1", group=r["group"], z_g90=0, z_a90=0, z_mins=z_mins, z_onoff=0, z_cs=0,
                                 age=age, team_ppg=team.get(y, team.mean()),
                                 eu_z_g90=0, eu_z_a90=0, eu_z_mins=-3.0 if te != "none" else 0.0,
                                 in_cl=int(te == "CL"), in_el=int(te == "EL"), trophy_pts=0.0))
        if not rows:
            out.append(dict(piv_mean_eur=np.nan, piv_lo_eur=np.nan, piv_hi_eur=np.nan, piv_by_season={},
                            piv_prime_by_season={},
                            value_delivered_eur=0.0))
            continue
        from .market import eu_role_columns
        df = eu_role_columns(pd.DataFrame(rows))
        lp = market_model.predict(df).values
        # age-neutral version for the deal balance: what that season's output is worth from a player in his prime
        lp_prime = market_model.predict(df.assign(age=PRIME_AGE)).values
        m = lp.mean()
        per_season = dict(zip(df["season"].astype(int), np.exp(lp)))
        n_all, n_cov = len(r["barca_seasons"]), len(rows)
        out.append(dict(
            piv_mean_eur=float(np.exp(m)),
            piv_lo_eur=float(np.exp(m - Z80 * sigma)), piv_hi_eur=float(np.exp(m + Z80 * sigma)),
            piv_by_season=per_season,
            piv_prime_by_season=dict(zip(df["season"].astype(int), np.exp(lp_prime))),
            # euro-seasons of value delivered; pre-2012 seasons credited at the covered average
            value_delivered_eur=float(np.exp(lp).sum() * n_all / n_cov)))
    o = pd.DataFrame(out, index=s.index)
    s = pd.concat([s, o], axis=1)
    fee_adj_eur = s["fee_adj"] * (1e6 if s["fee_adj"].max() < 1e5 else 1)
    for c in ("mean", "lo", "hi"):
        s[f"fair_fee_{c}_eur"] = s[f"piv_{c}_eur"] * premium
    s["fee_to_fair"] = fee_adj_eur / s["fair_fee_mean_eur"]
    # how far the fee sits from the fair fee, in standard deviations of the market model (log scale)
    s["fee_z"] = (np.log(fee_adj_eur) - np.log(s["fair_fee_mean_eur"])) / sigma
    s["market_verdict"] = np.select(
        [s["piv_mean_eur"].isna(), fee_adj_eur > s["fair_fee_hi_eur"], fee_adj_eur < s["fair_fee_lo_eur"]],
        ["No league minutes", "Overpaid", "Bargain"], "Fair price")
    return s


def load_dob():
    from .data import load
    p = load("players")[["player_id", "date_of_birth"]]
    return dict(zip(p["player_id"], p["date_of_birth"]))


VALUE_FORMULA = "{y} ~ value_delivered_m"


def club_verdicts(df: pd.DataFrame, y: str, alpha: float = 0.2) -> pd.DataFrame:
    """Club benchmark: net cost vs value delivered across Barça's own signings, leave-one-out 80% PI."""
    d = df.copy()
    d["value_delivered_m"] = d["value_delivered_eur"] / 1e6
    fair, lo, hi = [], [], []
    for i in d.index:
        m = smf.ols(VALUE_FORMULA.format(y=y), data=d.drop(index=i)).fit()
        p = m.get_prediction(d.loc[[i]]).summary_frame(alpha=alpha)
        fair.append(p["mean"].iloc[0]); lo.append(p["obs_ci_lower"].iloc[0]); hi.append(p["obs_ci_upper"].iloc[0])
    d[f"{y}_fair"], d[f"{y}_lo"], d[f"{y}_hi"] = fair, lo, hi
    d[f"{y}_resid"] = d[y] - d[f"{y}_fair"]
    d[f"{y}_verdict"] = np.select([d[y] > d[f"{y}_hi"], d[y] < d[f"{y}_lo"]], ["Overpaid", "Bargain"], "Fair price")
    return d


# ---------------------------------------------------------------------------------------------
# Deal balance: performance value + money back - fee
# ---------------------------------------------------------------------------------------------
USE_SEASONS = 5   # a fairly priced player "pays back" his fair fee over a typical 5-season contract


def add_deal_balance(d: pd.DataFrame, market_model, premium: float, use_seasons: int = USE_SEASONS) -> pd.DataFrame:
    """perf_value = sum over Barça seasons of (that season's age-neutral fair fee / use_seasons)
    Age-neutral: each season's output is priced as if from a player of PRIME_AGE, because what a season is *worth
    to Barça on the pitch* doesn't depend on birth year. Age still matters, but through the money back: young
    players keep resale value (sale fee or current value), old ones don't. Using age here too would count it twice.
    balance     = perf_value + recovered (sale + loan fees + current value if still at the club) - fee

    Reference point: paying exactly the fair fee, playing at that level for `use_seasons` seasons and leaving for
    free gives a balance of 0. The 80% range comes from the market model's error on the performance value."""
    k = np.exp(Z80 * np.sqrt(market_model.scale))
    d = d.copy()
    perf = []
    for _, r in d.iterrows():
        piv = r["piv_prime_by_season"] or {}
        n_all, n_cov = len(r["barca_seasons"]), len(piv)
        total = sum(piv.values()) * premium / 1e6 / use_seasons
        perf.append(total * n_all / n_cov if n_cov else 0.0)
    d["perf_value_adj"] = perf
    d["balance_adj"] = d["perf_value_adj"] + d["recovered_adj"] - d["fee_adj"]
    d["balance_lo"] = d["perf_value_adj"] / k + d["recovered_adj"] - d["fee_adj"]
    d["balance_hi"] = d["perf_value_adj"] * k + d["recovered_adj"] - d["fee_adj"]
    d["balance_verdict"] = np.select([d["balance_lo"] > 0, d["balance_hi"] < 0], ["Profit", "Loss"], "Break-even")
    # where the balance comes from, for the headline wording
    d["balance_driver"] = np.where(d["recovered_adj"] >= d["perf_value_adj"], "resale", "performance")
    return d


# ---------------------------------------------------------------------------------------------
# Decision quality: was the fee reasonable given what was known WHEN he signed?
# ---------------------------------------------------------------------------------------------
PRE_SEASONS = 2
PRE_MIN_MINUTES = 450


def add_pre_signing(d: pd.DataFrame, scored: pd.DataFrame, market_model, premium: float, index_df,
                    club_id: int = BARCA_ID) -> pd.DataFrame:
    """Two views of the fee at the moment of signing:
    1. Stats view: the market model applied to his last PRE_SEASONS league seasons BEFORE joining (any of the
       14 leagues in the data, minutes-weighted), times the usual premium, with an 80% interval.
    2. Market view: Transfermarkt's valuation on the signing date, times the usual premium, with the 10th-90th
       percentile of what clubs actually pay over valuations as the "normal" range.
    The decision verdict uses the stats view when he has pre-signing data, otherwise the market view."""
    from .market import prepare, eu_role_columns, market_premium_range
    from .inflation import adjuster
    adj = adjuster(index_df)
    sigma = np.sqrt(market_model.scale)
    p = prepare(scored)
    p = p[p["club_id"] != club_id]
    lo_q, hi_q = market_premium_range()
    out = []
    for _, r in d.iterrows():
        winter = pd.Timestamp(r["signing_date"]).month in (1, 2)
        last = r["signing_season"] if winter else r["signing_season"] - 1
        c = p[(p["player_id"] == r["player_id"]) & (p["season"] <= last) & (p["season"] >= last - 3)
              & (p["minutes"] >= PRE_MIN_MINUTES)]
        seasons = sorted(c["season"].unique())[-PRE_SEASONS:]
        c = c[c["season"].isin(seasons)]
        row = {}
        if len(c):
            lp = market_model.predict(eu_role_columns(c.copy())).values
            m = np.average(lp, weights=c["minutes"].values)
            row.update(pre_fair_eur=np.exp(m) * premium, pre_lo_eur=np.exp(m - Z80 * sigma) * premium,
                       pre_hi_eur=np.exp(m + Z80 * sigma) * premium,
                       pre_seasons=", ".join(f"{int(y)}/{str(int(y) + 1)[2:]}" for y in seasons),
                       pre_clubs=", ".join(sorted(set(c["league"]))),
                       pre_minutes=int(c["minutes"].sum()), pre_goals=int(c["goals"].sum()),
                       pre_assists=int(c["assists"].sum()), pre_avg_score=float(np.average(c["score"], weights=c["minutes"])))
        mv = r["mv_at_signing_eur"]
        if pd.notna(mv) and mv > 0:
            mv_adj = mv / adj(r["signing_season"])
            row.update(tm_fair_eur=mv_adj * premium, tm_lo_eur=mv_adj * lo_q, tm_hi_eur=mv_adj * hi_q, mv_sign_adj_eur=mv_adj)
        out.append(row)
    o = pd.DataFrame(out, index=d.index)
    for col in ["pre_fair_eur", "pre_lo_eur", "pre_hi_eur", "tm_fair_eur", "tm_lo_eur", "tm_hi_eur", "mv_sign_adj_eur",
                "pre_seasons", "pre_clubs", "pre_minutes", "pre_goals", "pre_assists", "pre_avg_score"]:
        if col not in o:
            o[col] = np.nan
    d = pd.concat([d, o], axis=1)
    fee = d["fee_adj"] * 1e6

    def verdict(f, lo, hi):
        return np.select([lo.isna(), f > hi, f < lo], ["No data", "Overpaid", "Bargain"], "Fair price")
    d["pre_verdict"] = verdict(fee, d["pre_lo_eur"], d["pre_hi_eur"])
    d["tm_verdict"] = verdict(fee, d["tm_lo_eur"], d["tm_hi_eur"])
    d["pre_ratio"] = fee / d["pre_fair_eur"]
    d["tm_ratio"] = fee / d["tm_fair_eur"]
    # Combine: stats see past performance but miss potential; the market valuation prices potential.
    # Overpaid / bargain only when both views agree (or when only one view exists).
    dec, src, note = [], [], []
    for a, b in zip(d["pre_verdict"], d["tm_verdict"]):
        if a == "No data" and b == "No data":
            dec.append("No data"); src.append("none"); note.append("")
        elif a == "No data" or b == "No data":
            v = b if a == "No data" else a
            dec.append(v); src.append("market value only" if a == "No data" else "stats only"); note.append("")
        elif a == b:
            dec.append(a); src.append("stats + market value"); note.append("")
        else:
            dec.append("Fair price"); src.append("stats + market value")
            note.append(f"stats: {a.lower()}, market value: {b.lower()}")
    d["decision"], d["decision_source"], d["decision_note"] = dec, src, note
    d["decision_ratio"] = np.where(d["pre_ratio"].notna(), d["pre_ratio"], d["tm_ratio"])
    d["decision_outcome"] = [quadrant(a, b) for a, b in zip(d["decision"], d["verdict"])]
    return d


def quadrant(decision, verdict):
    """Decision (at signing) x outcome (headline verdict, incl. 'leaning')."""
    if decision == "No data":
        return "No pre-signing data"
    v = verdict or ""
    outcome = ("good" if v.startswith(("Good", "Leaning good")) or v == "Never played, sold at a profit"
               else "bad" if v.startswith(("Bad", "Leaning bad")) or v == "Never played" else "fair")
    if decision in ("Bargain", "Fair price"):
        return {"good": "Smart buy, paid off", "fair": "Sound buy, broke even", "bad": "Sound buy, went wrong"}[outcome]
    return {"good": "Overpaid, but it paid off", "fair": "Overpaid, broke even",
            "bad": "Overpaid, as the numbers warned"}[outcome]
