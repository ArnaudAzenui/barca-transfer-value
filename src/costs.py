"""Cost side: inflation-adjusted fee, wages and money recovered, per signing (in BASE_SEASON euros)."""
import numpy as np
import pandas as pd
from .config import CURATED, BASE_SEASON
from .inflation import adjuster


_WAGE_FIT = {}


def add_costs(sign: pd.DataFrame, index_df: pd.DataFrame, club_id: int = 131, shares: pd.DataFrame = None) -> pd.DataFrame:
    """shares: output of wages.payroll_shares (Capology payroll split across each squad). When given, wages come
    from it season by season; the old estimates are kept in `wages_old_adj` for comparison."""
    adj = adjuster(index_df)
    s = sign.copy()
    s["fee_adj"] = s["fee_eur"] / s["signing_season"].map(adj)
    exit_s = s["exit_season"].fillna(BASE_SEASON).astype(int)
    s["sale_adj"] = s["sale_fee_eur"] / exit_s.map(adj)
    s["loan_income_adj"] = s["loan_income_eur"] / exit_s.map(adj)
    # still at the club: count his current market value as money Barça could recover (flagged in outputs)
    s["residual_value_adj"] = np.where(s["ongoing"], s["mv_latest_eur"].fillna(0) / adj(BASE_SEASON), 0.0)
    s["recovered_adj"] = s["sale_adj"] + s["loan_income_adj"] + s["residual_value_adj"]
    s["net_fee_cost_adj"] = s["fee_adj"] - s["recovered_adj"]

    # wages: reported estimates where we have them, otherwise imputed from market value at signing
    w = pd.read_csv(CURATED / "wages_reported.csv", comment="#")[["player", "gross_wage_eur_m"]]
    s = s.merge(w, on="player", how="left")
    # the imputation line is always fitted on Barça's reported wages; other clubs' players are all imputed
    if club_id != 131:
        s["gross_wage_eur_m"] = np.nan
    s["mv_sign_adj"] = s["mv_at_signing_eur"] / s["signing_season"].map(adj)
    known = s.dropna(subset=["gross_wage_eur_m", "mv_sign_adj"])
    known = known[known["mv_sign_adj"] > 0]
    if len(known) >= 5:
        _WAGE_FIT[index_df.attrs.get("basis", "football")] = np.polyfit(
            np.log(known["mv_sign_adj"]), np.log(known["gross_wage_eur_m"] * 1e6), 1)
    b, a = _WAGE_FIT[index_df.attrs.get("basis", "football")]   # run Barça first
    imputed = np.exp(a + b * np.log(s["mv_sign_adj"].clip(lower=1e6)))
    s["wage_source"] = np.where(s["gross_wage_eur_m"].notna(), "reported estimate", "imputed")
    s["gross_wage_eur"] = s["gross_wage_eur_m"].mul(1e6).fillna(imputed)
    s["wages_old_adj"] = [sum(w / adj(y) for y in yrs) for w, yrs in zip(s["gross_wage_eur"], s["barca_seasons"])]
    s["wages_adj"] = s["wages_old_adj"]
    if shares is not None:
        sh = shares[shares["club_id"] == club_id].set_index(["player_id", "season"])
        wages, srcs, per = [], [], []
        for _, r in s.iterrows():
            by = {y: float(sh.loc[(r["player_id"], y), "wage_eur"]) for y in r["barca_seasons"] if (r["player_id"], y) in sh.index}
            fill = np.mean(list(by.values())) if by else r["gross_wage_eur"]
            yrs = {y: by.get(y, fill) for y in r["barca_seasons"]}
            wages.append(sum(v / adj(y) for y, v in yrs.items()))
            per.append(yrs)
            ext = any(bool(sh.loc[(r["player_id"], y), "extrapolated"]) for y in by)
            kinds = {sh.loc[(r["player_id"], y), "wage_kind"] for y in by}
            label = ("Capology actual + payroll share" if "Capology actual" in kinds
                     else "Capology payroll share, player-adjusted" if "payroll share, player-adjusted" in kinds
                     else "Capology payroll share")
            srcs.append(label + (" (pre-2013 extrapolated)" if ext else "") if by else r["wage_source"])
        s["wages_adj"], s["wage_source"], s["wage_by_season"] = wages, srcs, per
    s["net_total_cost_adj"] = s["net_fee_cost_adj"] + s["wages_adj"]
    return s.drop(columns=["gross_wage_eur_m"])
