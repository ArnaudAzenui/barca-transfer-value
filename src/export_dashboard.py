"""Export the dashboard data (outputs/dashboard_data.json): one block per money basis."""
import json
import numpy as np
import pandas as pd
from .config import PROCESSED, OUTPUTS, BARCA_ID
from .inflation import build_index, BASES
from .value import USE_SEASONS as USE
from .market import market_premium


def r(x, n=2):
    if x is None:
        return None
    try:
        if np.isnan(x):
            return None
    except TypeError:
        pass
    return round(float(x), n)


BASIS_FIELDS = lambda x: dict(
    fee=r(x["fee_adj"]), sale=r(x["sale_adj"]), loans=r(x["loan_income_adj"]), residual=r(x["residual_value_adj"]),
    recovered=r(x["recovered_adj"]), wages=r(x["wages_adj"]), net_total=r(x["net_total_cost_adj"]),
    fair=r(x["fair_fee_mean_eur"] / 1e6), fair_lo=r(x["fair_fee_lo_eur"] / 1e6), fair_hi=r(x["fair_fee_hi_eur"] / 1e6),
    ratio=r(x["fee_to_fair"]), market=x["market_verdict"],
    value=r(x["value_delivered_eur"] / 1e6, 1), club_fair=r(x["net_total_cost_adj_fair"]),
    club_lo=r(x["net_total_cost_adj_lo"]), club_hi=r(x["net_total_cost_adj_hi"]), club=x["net_total_cost_adj_verdict"],
    perf=r(x["perf_value_adj"]), balance=r(x["balance_adj"]), bal_lo=r(x["balance_lo"]), bal_hi=r(x["balance_hi"]),
    bal=x["balance_verdict"], verdict=x["verdict"],
    pre_fair=r(x["pre_fair_eur"] / 1e6), pre_lo=r(x["pre_lo_eur"] / 1e6), pre_hi=r(x["pre_hi_eur"] / 1e6),
    pre_ratio=r(x["pre_ratio"]), pre_verdict=x["pre_verdict"], tm_fair=r(x["tm_fair_eur"] / 1e6),
    tm_lo=r(x["tm_lo_eur"] / 1e6), tm_hi=r(x["tm_hi_eur"] / 1e6), tm_ratio=r(x["tm_ratio"]), tm_verdict=x["tm_verdict"],
    decision=x["decision"], decision_note=x["decision_note"] or "", decision_outcome=x["decision_outcome"],
    decision_ratio=r(x["decision_ratio"]),
    # per-season credit used in the deal balance (age-neutral fair value / use_seasons); sums to `perf`
    piv={int(k): r(v * PREM / 1e6 / USE) for k, v in (x["piv_prime_by_season"] or {}).items()})


PREM = None
LEAGUE_NAMES = {"ES1": "La Liga", "GB1": "Premier League", "IT1": "Serie A", "L1": "Bundesliga", "FR1": "Ligue 1",
                "PO1": "Liga Portugal", "NL1": "Eredivisie", "BE1": "Belgian Pro League", "TR1": "Süper Lig",
                "RU1": "Russian Premier Liga", "GR1": "Greek Super League", "UKR1": "Ukrainian Premier Liga",
                "SC1": "Scottish Premiership", "DK1": "Danish Superliga"}


def export():
    global PREM
    PREM = market_premium()
    res = pd.read_pickle(PROCESSED / "results_by_basis.pkl")
    scored = pd.read_pickle(PROCESSED / "player_seasons.pkl")
    b = scored[scored["club_id"] == BARCA_ID].set_index(["player_id", "season"])
    summ = json.loads((OUTPUTS / "model_summary.json").read_text())
    ix = build_index("football")
    base = res["football"]
    players = []
    for i, x in base.iterrows():
        seasons = [int(y) for y in x["barca_seasons"]]
        per_season = {}
        for y in seasons:
            if (x["player_id"], y) in b.index:
                s = b.loc[(x["player_id"], y)]
                per_season[y] = dict(lm=int(s["minutes"]), lg=int(s["goals"]), la=int(s["assists"]), ls=r(s["score"]),
                                     em=int(s["eu_minutes"]), eg=int(s["eu_goals"]), ea=int(s["eu_assists"]),
                                     es=r(s["eu_score"]), comp=s["eu_comps"] if isinstance(s["eu_comps"], str) else "",
                                     tp=r(s["trophy_pts"]), tr=s["trophies"])
        players.append(dict(
            name=x["player"], group=x["group"], pos=x["sub_position"] if isinstance(x["sub_position"], str) else x["group"],
            season=int(x["signing_season"]), from_club=x["from_club"], age=r(x["age_at_signing"], 1),
            ongoing=bool(x["ongoing"]), exit_to=x["exit_to"] if isinstance(x["exit_to"], str) else None,
            fee_nom=r(x["fee_eur"] / 1e6), sale_nom=r(x["sale_fee_eur"] / 1e6), wage_src=x["wage_source"],
            seasons=seasons, covered=int(x["seasons_covered"]),
            minutes=int(x["liga_minutes"]), goals=int(x["liga_goals"]), assists=int(x["liga_assists"]),
            eu_minutes=int(x["eu_minutes"]), eu_goals=int(x["eu_goals"]), eu_assists=int(x["eu_assists"]),
            avg_score=r(x["avg_score"]), avg_eu_score=r(x["avg_eu_score"]), trophy_pts=r(x["trophy_pts"]),
            per_season=per_season, source=x["source"], confidence=x["confidence"],
            pre_seasons=x["pre_seasons"] if isinstance(x["pre_seasons"], str) else None,
            pre_leagues=", ".join(LEAGUE_NAMES.get(c.strip(), c.strip()) for c in x["pre_clubs"].split(","))
                        if isinstance(x["pre_clubs"], str) else None,
            pre_minutes=None if pd.isna(x["pre_minutes"]) else int(x["pre_minutes"]),
            pre_goals=None if pd.isna(x["pre_goals"]) else int(x["pre_goals"]),
            pre_assists=None if pd.isna(x["pre_assists"]) else int(x["pre_assists"]),
            by={bs: BASIS_FIELDS(res[bs].loc[i]) for bs in res}))
    from .config import TROPHY_WEIGHTS, TROPHY_NAMES
    allc = pd.read_pickle(PROCESSED / "results_all_clubs.pkl")
    rivals = [dict(club=x["club"], name=x["player"], season=int(x["signing_season"]), pos=x["group"],
                   from_club=x["from_club"], fee=r(x["fee_adj"]), fee_nom=r(x["fee_eur"] / 1e6),
                   balance=r(x["balance_adj"]), bal_lo=r(x["balance_lo"]), bal_hi=r(x["balance_hi"]),
                   verdict=x["verdict"], decision=x["decision"], outcome=x["decision_outcome"],
                   ratio=r(x["decision_ratio"]), tm_ratio=r(x["tm_ratio"]), exit_to=x["exit_to"] if isinstance(x["exit_to"], str) else None,
                   ongoing=bool(x["ongoing"]))
              for _, x in allc.iterrows()]
    out = dict(summary=summ, bases=BASES, rivals=rivals, trophy_weights={TROPHY_NAMES[k]: v for k, v in TROPHY_WEIGHTS.items()},
               trophy_beta=summ["headline"].get("trophy_beta"),
               index=[dict(season=int(s), football=r(f, 3), cpi=r(c, 3)) for s, f, c in
                      zip(ix["season"], ix["football_index"], ix["cpi_index"])],
               players=players)
    (OUTPUTS / "dashboard_data.json").write_text(json.dumps(out, ensure_ascii=False))
    return out


if __name__ == "__main__":
    o = export(); print(len(o["players"]), "players")
