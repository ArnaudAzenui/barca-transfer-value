"""Run everything: python -m src.pipeline  (from the project root).

The analysis is run once per money basis (football index, consumer prices, actual euros). The football index is the
headline; the other two show how much the verdicts depend on the inflation adjustment.
"""
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from .config import PROCESSED, OUTPUTS
from .inflation import build_index, BASES
from .signings import build_signings
from .performance import player_seasons, score
from .europe import add_europe
from .trophies import add_trophies
from .wages import payroll_shares, validate as validate_wages
from .costs import add_costs
from .market import fit_market, market_premium
from .value import add_performance, add_piv, club_verdicts, add_deal_balance, add_pre_signing, VALUE_FORMULA, USE_SEASONS
from .config import BARCA_ID

RIVALS = {418: "Real Madrid", 13: "Atlético Madrid"}
CLUBS = {BARCA_ID: "Barcelona", **RIVALS}

M = 1e6
HEADLINE = "football"


def headline(balance, market, club, perf_value, fee):
    """Headline verdict = the deal balance (performance value + money back - fee).
    When the balance is inside its uncertainty range, the other two tests break the tie."""
    if market == "No league minutes":
        return "Never played" + (", sold at a profit" if balance == "Profit" else "")
    if balance == "Profit":
        base = "Good deal" + (" (resale)" if perf_value < fee else "")
    elif balance == "Loss":
        base = "Bad deal"
    else:
        base = "Fair deal"
        if market == club == "Overpaid":
            return "Leaning bad deal"
        if market == club == "Bargain":
            return "Leaning good deal"
        return base
    if base.startswith("Bad") and market == club == "Overpaid":
        return "Bad deal (all tests agree)"
    if base == "Good deal" and market == club == "Bargain":
        return "Good deal (all tests agree)"
    return base


def build_scored(use_cache=False):
    f = PROCESSED / "player_seasons.pkl"
    if use_cache and f.exists():
        return pd.read_pickle(f)
    scored = add_trophies(add_europe(score(player_seasons())))
    scored.to_pickle(f)
    return scored


SHARES = {}


def evaluate_club(club_id, sign, scored, prem, ix, mm, mm_all):
    if "w" not in SHARES:
        # Capology tables are optional (git-ignored); without them wages fall back to wages_reported.csv
        from .config import CURATED
        SHARES["w"] = payroll_shares(scored) if (CURATED / "capology_payrolls.csv").exists() else None
    d = add_performance(add_costs(sign, ix, club_id, SHARES["w"]), scored, club_id)
    d["wages_old_adj"] = d["wages_old_adj"] / M
    for c in ["fee_adj", "sale_adj", "loan_income_adj", "residual_value_adj", "recovered_adj",
              "net_fee_cost_adj", "wages_adj", "net_total_cost_adj"]:
        d[c] = d[c] / M
    d = add_piv(d, scored, mm, premium=prem, club_id=club_id)
    d = club_verdicts(d, "net_total_cost_adj")
    d = add_deal_balance(d, mm, prem)
    d["verdict"] = [headline(a, b, c, pv, f) for a, b, c, pv, f in
                    zip(d["balance_verdict"], d["market_verdict"], d["net_total_cost_adj_verdict"],
                        d["perf_value_adj"], d["fee_adj"])]
    d = add_pre_signing(d, scored, mm_all, prem, ix, club_id)
    # spells with no usable appearance data at all (e.g. Atlético 2014/15 line-ups are missing from the dataset)
    nodata = (d["seasons"] > 0) & (d["seasons_covered"] == 0) & (d["signing_season"] >= 2012)
    d.loc[nodata, ["verdict", "decision_outcome"]] = "No performance data"
    d.loc[nodata, ["perf_value_adj", "balance_adj", "balance_lo", "balance_hi"]] = np.nan
    d.loc[nodata, "balance_verdict"] = "No data"
    d["partial_data"] = d["seasons_covered"] < d["seasons"]
    d["club"] = CLUBS[club_id]
    return d


def run_basis(basis, sign, scored, prem):
    ix = build_index(basis)
    mm, market_rows = fit_market(scored, ix)
    mm_all, _ = fit_market(scored, ix, leagues=None)
    d = evaluate_club(BARCA_ID, sign, scored, prem, ix, mm, mm_all)
    d["basis"] = basis
    club_fit = smf.ols(VALUE_FORMULA.format(y="net_total_cost_adj"), data=d).fit()
    summary = {
        "basis": basis, "basis_label": BASES[basis],
        "market_model": {"n": int(mm.nobs), "r2": round(mm.rsquared, 3), "sigma_log": round(float(np.sqrt(mm.scale)), 3)},
        "market_model_all_leagues": {"n": int(mm_all.nobs), "r2": round(mm_all.rsquared, 3),
                                     "sigma_log": round(float(np.sqrt(mm_all.scale)), 3)},
        "market_premium": round(prem, 3), "use_seasons": USE_SEASONS,
        "trophy_beta": round(float(mm.params["trophy_pts"]), 4), "trophy_beta_se": round(float(mm.bse["trophy_pts"]), 4),
        "club_model": {"n": int(club_fit.nobs), "r2": round(club_fit.rsquared, 3),
                       "intercept_m": round(club_fit.params["Intercept"], 2),
                       "slope": round(club_fit.params["value_delivered_m"], 3)},
    }
    d.attrs["mm_all"] = None
    MODELS[basis] = mm_all
    return d, summary, mm, market_rows, ix


MODELS = {}


def premium_tests(allclubs: pd.DataFrame) -> dict:
    """Does Barça pay more than its rivals for comparable players? EUR10m+ deals, controlling for signing year
    (and age for the stats view). Coefficients are log-ratios relative to Atlético."""
    out = {}
    x = allclubs[(allclubs["fee_eur"] >= 10e6) & (allclubs["tm_ratio"] > 0) & np.isfinite(allclubs["tm_ratio"])].copy()
    x["lr"] = np.log(x["tm_ratio"])
    views = {"vs_market_value": ("lr ~ C(club, Treatment('Atlético Madrid')) + signing_season", x)}
    y = allclubs[(allclubs["fee_eur"] >= 10e6) & allclubs["pre_ratio"].notna()].copy()
    y["lr"] = np.log(y["pre_ratio"])
    views["vs_stats_fair_fee"] = ("lr ~ C(club, Treatment('Atlético Madrid')) + signing_season + age_at_signing", y)
    for k, (f, data) in views.items():
        m = smf.ols(f, data=data).fit()
        res = {"n": int(m.nobs)}
        for club in ("Barcelona", "Real Madrid"):
            term = f"C(club, Treatment('Atlético Madrid'))[T.{club}]"
            res[club] = {"premium_pct": round((np.exp(m.params[term]) - 1) * 100, 1),
                         "ci_pct": [round((np.exp(v) - 1) * 100, 1) for v in m.conf_int().loc[term]],
                         "p": round(float(m.pvalues[term]), 3)}
        out[k] = res
    return out


def club_summary(d: pd.DataFrame) -> dict:
    """Club-level transfer record (football €). Spells without performance data are left out."""
    d = d[d["verdict"] != "No performance data"]
    dec = d[d["decision"] != "No data"]
    good = d["verdict"].str.startswith(("Good", "Leaning good", "Never played, sold"))
    bad = d["verdict"].str.startswith(("Bad", "Leaning bad")) | (d["verdict"] == "Never played")
    return {
        "signings": int(len(d)), "spent_m": round(float(d["fee_adj"].sum()), 1),
        "recovered_m": round(float(d["recovered_adj"].sum()), 1),
        "perf_value_m": round(float(d["perf_value_adj"].sum()), 1),
        "total_balance_m": round(float(d["balance_adj"].sum()), 1),
        "wages_m": round(float(d["wages_adj"].sum()), 1),
        "return_per_eur_total_cost": round(float((d["perf_value_adj"].sum() + d["recovered_adj"].sum())
                                                 / (d["fee_adj"].sum() + d["wages_adj"].sum())), 3),
        "balance_per_eur": round(float(d["balance_adj"].sum() / d["fee_adj"].sum()), 3),
        "median_balance_m": round(float(d["balance_adj"].median()), 1),
        "good_share": round(float(good.mean()), 3), "bad_share": round(float(bad.mean()), 3),
        "big_money": int((d["fee_adj"] >= 60).sum()),
        "big_money_bad": int(((d["fee_adj"] >= 60) & bad).sum()),
        "decisions_rated": int(len(dec)),
        "overpaid_at_signing_share": round(float((dec["decision"] == "Overpaid").mean()), 3) if len(dec) else None,
        "median_fee_to_fair_at_signing": round(float(dec["decision_ratio"].median()), 3) if len(dec) else None,
        "median_fee_to_tm_value": round(float(d["tm_ratio"].median()), 3),
    }


def run(verbose=True, use_cache=False):
    PROCESSED.mkdir(parents=True, exist_ok=True); OUTPUTS.mkdir(parents=True, exist_ok=True)
    sign = build_signings()
    scored = build_scored(use_cache)
    prem = market_premium()
    results, summaries = {}, {}
    for basis in BASES:
        d, s, mm, market_rows, ix = run_basis(basis, sign, scored, prem)
        results[basis], summaries[basis] = d, s
        if basis == HEADLINE:
            head = (d, s, mm, market_rows, ix)
    d, s, mm, market_rows, ix = head
    s["n_signings"] = int(len(d))

    # rivals: same method, football-€ basis
    rivals = {}
    for cid, name in RIVALS.items():
        rivals[cid] = evaluate_club(cid, build_signings(cid), scored, prem, ix, mm, MODELS[HEADLINE])
    allclubs = pd.concat([d] + list(rivals.values()), ignore_index=True)
    clubs = {CLUBS[BARCA_ID]: club_summary(d), **{CLUBS[c]: club_summary(x) for c, x in rivals.items()}}
    # same comparison restricted to 2013/14+ (fee and appearance data are complete for all three clubs)
    clubs_2013 = {n: club_summary(x[x["signing_season"] >= 2013]) for n, x in allclubs.groupby("club")}
    s["clubs"] = clubs; s["clubs_2013"] = clubs_2013
    s["premium_tests"] = premium_tests(allclubs)
    if SHARES["w"] is not None:
        s["wage_curve"] = SHARES["w"].attrs["fit"]
        s["wage_validation"] = validate_wages(scored).to_dict(orient="records")
    pd.to_pickle(allclubs, PROCESSED / "results_all_clubs.pkl")

    build_index(HEADLINE).to_csv(OUTPUTS / "inflation_index.csv", index=False)
    pd.to_pickle(results, PROCESSED / "results_by_basis.pkl")
    d.to_pickle(PROCESSED / "results.pkl")

    cols = ["player", "group", "sub_position", "signing_season", "from_club", "age_at_signing", "ongoing",
            "exit_to", "seasons", "liga_minutes", "liga_goals", "liga_assists", "eu_minutes", "eu_goals", "eu_assists",
            "avg_score", "avg_eu_score", "trophy_pts", "trophies", "fee_eur", "fee_adj", "piv_mean_eur", "fair_fee_mean_eur", "fair_fee_lo_eur",
            "fair_fee_hi_eur", "fee_to_fair", "market_verdict", "perf_value_adj", "recovered_adj", "balance_adj",
            "balance_lo", "balance_hi", "balance_verdict", "wages_adj", "wage_source",
            "net_total_cost_adj", "value_delivered_eur", "net_total_cost_adj_fair", "net_total_cost_adj_verdict",
            "verdict", "partial_data", "seasons_covered", "source", "confidence"]
    d[cols].to_csv(OUTPUTS / "barca_signings_verdicts.csv", index=False)
    # one verdict table across money bases
    vb = pd.DataFrame({"player": d["player"], "signing_season": d["signing_season"], "fee_actual_m": d["fee_eur"] / M})
    for b, x in results.items():
        vb[f"fee_{b}_m"] = x["fee_adj"].values
        vb[f"verdict_{b}"] = x["verdict"].values
    vb.to_csv(OUTPUTS / "verdicts_by_money_basis.csv", index=False)
    ac_cols = ["club", "player", "group", "signing_season", "from_club", "age_at_signing", "fee_eur", "fee_adj",
               "pre_fair_eur", "pre_ratio", "pre_verdict", "tm_fair_eur", "tm_ratio", "tm_verdict", "decision",
               "decision_source", "decision_note", "perf_value_adj", "recovered_adj", "balance_adj", "balance_lo", "balance_hi",
               "balance_verdict", "verdict", "decision_outcome", "trophy_pts", "exit_to", "ongoing"]
    allclubs[ac_cols].to_csv(OUTPUTS / "signings_barca_real_atletico.csv", index=False)
    d[cols + ["pre_fair_eur", "pre_lo_eur", "pre_hi_eur", "pre_ratio", "pre_verdict", "pre_seasons", "tm_fair_eur",
              "tm_ratio", "tm_verdict", "decision", "decision_source", "decision_note", "decision_outcome"]].to_csv(
        OUTPUTS / "barca_signings_verdicts.csv", index=False)
    (OUTPUTS / "model_summary.json").write_text(json.dumps({"headline": s, "by_basis": summaries}, indent=2, default=float))
    if verbose:
        print(json.dumps(summaries, indent=2))
    return d, s, mm, market_rows, ix, scored, results


if __name__ == "__main__":
    run()
