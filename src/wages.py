"""Player wages from Capology.

Inputs (data/curated/, extracted from pages saved in data/raw/capology/):
  capology_payrolls.csv                 club gross fixed payroll per season, 2013/14-2026/27 (Barça, Real, Atlético)
  capology_salaries_barca_history.csv   Barça player salaries for 2017/18, 2018/19, 2020/21
  capology_salaries_2026_27.csv         Real Madrid and Atlético player salaries for 2026/27

Method
  1. Wage curve: log(gross fixed salary) ~ log(market value that season) + age + club-season fixed effect, fitted on
     every Capology player salary we can match to the dataset. The fixed effect absorbs each club-season's pay level,
     so the curve only describes how pay is split *within* a squad.
  2. Payroll split: each club-season's Capology payroll is shared across its squad in proportion to the curve.
  3. Player correction: where Capology lists a player's actual salary, that figure is used; in his other seasons at
     the club, his split-based estimate is scaled by his own actual/estimated ratio (a player who was paid more than
     his value suggested in 2018 was likely paid more in 2019 too).
  Seasons before 2013/14 reuse the 2013/14 payroll (flagged as extrapolated).
"""
import unicodedata
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from .config import CURATED
from .data import load, season_of

CLUB_IDS = {"Barcelona": 131, "Real Madrid": 418, "Atlético Madrid": 13}
ALIASES = {"arthur": "arthur melo", "carles alena": "carles alena", "rafinha": "rafinha"}


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()
    return ALIASES.get(s, s)


def _season_values():
    """Latest valuation in each season; also carried forward one season (the dataset ends in July 2026, so
    2026/27 salaries are matched to end-of-2025/26 values)."""
    v = load("player_valuations")[["player_id", "date", "market_value_in_eur"]].copy()
    v["season"] = season_of(v["date"])
    mv = v.sort_values("date").groupby(["player_id", "season"])["market_value_in_eur"].last().rename("mv").reset_index()
    nxt = mv.assign(season=mv["season"] + 1)
    both = pd.concat([mv.assign(p=0), nxt.assign(p=1)]).sort_values("p")
    return both.drop_duplicates(["player_id", "season"])[["player_id", "season", "mv"]]


def _squads(scored):
    """La Liga squads; reconstructed pre-2012 seasons are only a handful of signings, not a squad."""
    s = scored[scored["league"] == "ES1"]
    return s[~s["reconstructed"].astype(bool)] if "reconstructed" in s else s


def actual_salaries(scored: pd.DataFrame) -> pd.DataFrame:
    """All Capology player salaries matched to dataset player_ids (by name within the club's squad that season)."""
    files = ["capology_salaries_barca_history.csv", "capology_salaries_2026_27.csv"]
    sal = pd.concat([pd.read_csv(CURATED / f) for f in files if (CURATED / f).exists()], ignore_index=True)
    sal["club_id"] = sal["club"].map(CLUB_IDS)
    sal["k"] = sal["player"].map(_norm)
    names = load("players")[["player_id", "name", "current_club_id", "date_of_birth"]].copy()
    names["k"] = names["name"].map(_norm)
    squads = _squads(scored)[["club_id", "season", "player_id"]].drop_duplicates()
    out = []
    for _, r in sal.iterrows():
        cand = names[names["k"] == r["k"]]
        if cand.empty:   # fall back to surname + first initial
            parts = r["k"].split()
            cand = names[names["k"].str.endswith(" " + parts[-1]) & names["k"].str.startswith(parts[0][0])] if len(parts) > 1 else cand
        if len(cand) > 1:  # disambiguate: in the club's squad that season, or currently at the club
            in_sq = cand[cand["player_id"].isin(squads[(squads["club_id"] == r["club_id"]) & (squads["season"] == r["season"])]["player_id"])]
            cand = in_sq if len(in_sq) else cand[cand["current_club_id"] == r["club_id"]]
        if len(cand) == 1:
            out.append(dict(club_id=r["club_id"], season=int(r["season"]), player_id=int(cand["player_id"].iloc[0]),
                            actual_wage_eur=r["gross_fixed_per_year_eur"], dob=cand["date_of_birth"].iloc[0]))
    a = pd.DataFrame(out).dropna(subset=["actual_wage_eur"])
    return a[a["actual_wage_eur"] > 0]


def fit_wage_curve(scored: pd.DataFrame, exclude_season=None):
    a = actual_salaries(scored).merge(_season_values(), on=["player_id", "season"], how="left")
    a = a[a["mv"] > 0]
    if exclude_season is not None:
        a = a[~((a["club_id"] == 131) & (a["season"] == exclude_season))]
    a["age"] = (pd.to_datetime((a["season"] + 1).astype(str) + "-01-01") - pd.to_datetime(a["dob"])).dt.days / 365.25
    a["lw"], a["lmv"] = np.log(a["actual_wage_eur"]), np.log(a["mv"])
    a["cs"] = a["club_id"].astype(str) + "_" + a["season"].astype(str)
    return smf.ols("lw ~ lmv + age + C(cs)", data=a).fit()


def _split(scored, fit):
    b_mv, b_age = fit.params["lmv"], fit.params["age"]
    pay = pd.read_csv(CURATED / "capology_payrolls.csv")
    pay["club_id"] = pay["club"].map(CLUB_IDS)
    sq = _squads(scored)
    sq = sq[sq["club_id"].isin(CLUB_IDS.values())][
        ["club_id", "season", "player_id", "date_of_birth"]].copy()
    sq = sq.merge(_season_values(), on=["player_id", "season"], how="left")
    sq["mv"] = sq["mv"].fillna(sq.groupby(["club_id", "season"])["mv"].transform("median")).clip(lower=1e5)
    sq["age"] = ((pd.to_datetime((sq["season"] + 1).astype(str) + "-01-01") - pd.to_datetime(sq["date_of_birth"])).dt.days / 365.25).fillna(26)
    sq["w"] = np.exp(b_mv * np.log(sq["mv"]) + b_age * sq["age"])
    sq["share"] = sq["w"] / sq.groupby(["club_id", "season"])["w"].transform("sum")
    first = pay.groupby("club_id")["season"].min().to_dict()
    sq["pay_season"] = [max(s, first.get(c, s)) for c, s in zip(sq["club_id"], sq["season"])]
    sq = sq.merge(pay[["club_id", "season", "gross_fixed_per_year_eur"]].rename(columns={"season": "pay_season"}),
                  on=["club_id", "pay_season"], how="left")
    sq["est_wage_eur"] = sq["share"] * sq["gross_fixed_per_year_eur"]
    sq["extrapolated"] = sq["pay_season"] != sq["season"]
    return sq


def validate(scored: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-season-out check on Barça's seasons with Capology player salaries: estimate that season from the
    payroll split with a curve fitted WITHOUT it, and compare with Capology's actual figures."""
    act = actual_salaries(scored)
    rows = []
    for s in sorted(act[act["club_id"] == 131]["season"].unique()):
        sq = _split(scored, fit_wage_curve(scored, exclude_season=s))
        m = act[(act["club_id"] == 131) & (act["season"] == s)].merge(sq, on=["club_id", "season", "player_id"])
        m["abs_pct_err"] = (m["est_wage_eur"] / m["actual_wage_eur"] - 1).abs()
        rows.append(dict(season=int(s), n=len(m), corr_log=round(float(np.corrcoef(np.log(m["est_wage_eur"]), np.log(m["actual_wage_eur"]))[0, 1]), 3),
                         median_abs_pct_err=round(float(m["abs_pct_err"].median()) * 100, 1)))
    return pd.DataFrame(rows)


def payroll_shares(scored: pd.DataFrame) -> pd.DataFrame:
    """Best wage estimate for every (club, season, player) in the three clubs' La Liga squads."""
    fit = fit_wage_curve(scored)
    sq = _split(scored, fit)
    act = actual_salaries(scored)
    sq = sq.merge(act[["club_id", "season", "player_id", "actual_wage_eur"]], on=["club_id", "season", "player_id"], how="left")
    # player-specific correction from seasons where his actual salary is known
    known = sq.dropna(subset=["actual_wage_eur"])
    lr = (np.log(known["actual_wage_eur"]) - np.log(known["est_wage_eur"])).groupby([known["club_id"], known["player_id"]])
    # one salary year is too thin to rescale a whole spell (e.g. de Jong's deferred 2020/21 wage): need two or more
    ratio = np.exp(lr.mean()[lr.size() >= 2]).rename("ratio").reset_index()
    sq = sq.merge(ratio, on=["club_id", "player_id"], how="left")
    sq["wage_eur"] = np.where(sq["actual_wage_eur"].notna(), sq["actual_wage_eur"],
                              sq["est_wage_eur"] * sq["ratio"].fillna(1.0))
    sq["wage_kind"] = np.select([sq["actual_wage_eur"].notna(), sq["ratio"].notna()],
                                ["Capology actual", "payroll share, player-adjusted"], "payroll share")
    out = sq[["club_id", "season", "player_id", "wage_eur", "est_wage_eur", "actual_wage_eur", "share", "wage_kind", "extrapolated"]]
    out.attrs["fit"] = {"b_mv": round(float(fit.params["lmv"]), 3), "b_age": round(float(fit.params["age"]), 3),
                        "r2_within": None, "n": int(fit.nobs), "n_actual_matched": int(len(act))}
    return out
