"""Money bases: how euros from different seasons are made comparable.

basis="football" (default, recommended)
    Average Transfermarkt value of the 200 most valuable players in the top-5 leagues each season,
    rescaled so BASE_SEASON = 1.0. Tracks the price of *footballers*, which rose ~3x from 2010 to 2025.
basis="cpi"
    Euro-area consumer prices (HICP, data/curated/euro_hicp.csv). Tracks the price of *everything else*;
    rose only ~1.4x over the same period, so it barely moves fees.
basis="nominal"
    No adjustment: the euros actually paid at the time.

Adjusted euros = nominal euros / index(season).

Why valuations and not fees for the football index? The open dataset only has transfer histories for players
still active after ~2012, so the pre-2013 fee record is patchy and a fee-based index is biased low. Where fee
coverage is good (2014+), the fee-based index tracks the valuation index closely (fee_index column).
"""
import numpy as np
import pandas as pd
from .config import TOP5, BASE_SEASON, FIRST_SEASON, CURATED
from .data import load, season_of

TOP_N_PLAYERS = 200
TOP_N_FEES = 100
BASES = {"football": "Today's football €", "cpi": "Today's € (consumer prices)", "nominal": "Actual € at the time"}


def _football() -> pd.DataFrame:
    v = load("player_valuations").copy()
    v["season"] = season_of(v["date"])
    v = v[v["player_club_domestic_competition_id"].isin(TOP5)]
    pv = v.groupby(["season", "player_id"])["market_value_in_eur"].max().reset_index()
    mv = (pv.sort_values("market_value_in_eur", ascending=False).groupby("season").head(TOP_N_PLAYERS)
            .groupby("season")["market_value_in_eur"].mean().rename("top200_mean_value"))
    t = load("transfers").copy()
    clubs = load("clubs")[["club_id", "domestic_competition_id"]]
    t = t.merge(clubs, left_on="to_club_id", right_on="club_id", how="left")
    t = t[t["domestic_competition_id"].isin(TOP5) & (t["transfer_fee"] > 0)]
    t["season"] = season_of(t["transfer_date"])
    fees = (t.sort_values("transfer_fee", ascending=False).groupby("season").head(TOP_N_FEES)
              .groupby("season")["transfer_fee"].mean().rename("top100_mean_fee"))
    ix = pd.concat([mv, fees], axis=1).loc[FIRST_SEASON - 2:BASE_SEASON]
    ix["football_index"] = ix["top200_mean_value"] / ix.loc[BASE_SEASON, "top200_mean_value"]
    ix["fee_index"] = ix["top100_mean_fee"] / ix.loc[BASE_SEASON, "top100_mean_fee"]
    ix.index.name = "season"
    return ix


def _cpi() -> pd.Series:
    c = pd.read_csv(CURATED / "euro_hicp.csv", comment="#").set_index("year")["hicp_rate_pct"]
    level = (1 + c / 100).cumprod()
    return (level / level.loc[BASE_SEASON]).rename("cpi_index")


def build_index(basis: str = "football") -> pd.DataFrame:
    """Returns a table with every index; the column `index` is the one for `basis`."""
    ix = _football().join(_cpi(), how="left")
    ix["nominal_index"] = 1.0
    ix["index"] = ix[f"{basis}_index"]
    ix.attrs["basis"] = basis
    return ix.reset_index()


def adjuster(index_df: pd.DataFrame):
    m = dict(zip(index_df["season"].astype(int), index_df["index"]))
    lo, hi = min(m), max(m)
    return lambda season: m[int(min(max(season, lo), hi))]
