"""Load the transfermarkt-datasets tables (github.com/dcaribou/transfermarkt-datasets)."""
from __future__ import annotations
import zipfile
from functools import lru_cache
from pathlib import Path
import pandas as pd
from .config import RAW, DATASET_URL

TABLES = ["players", "transfers", "appearances", "games", "player_valuations", "clubs", "game_lineups"]


def _find_zip() -> Path | None:
    from .config import ROOT
    cands = sorted(RAW.glob("transfermarkt-datasets*.zip")) or sorted(ROOT.glob("transfermarkt-datasets*.zip"))
    if not cands:
        dl = Path.home() / "Downloads"
        cands = sorted(dl.glob("transfermarkt-datasets*.zip")) if dl.exists() else []
    return cands[-1] if cands else None


def ensure_extracted() -> Path:
    """Unzip the dataset into data/raw/tm/ (downloads it first if nothing is there)."""
    out = RAW / "tm"
    if out.exists() and any(out.rglob("players.csv*")):
        return out
    z = _find_zip()
    if z is None:
        import urllib.request
        RAW.mkdir(parents=True, exist_ok=True)
        z = RAW / "transfermarkt-datasets.zip"
        print(f"Downloading {DATASET_URL} ...")
        urllib.request.urlretrieve(DATASET_URL, z)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(out)
    return out


def _path(name: str) -> Path:
    root = ensure_extracted()
    for pat in (f"{name}.csv.gz", f"{name}.csv"):
        hits = list(root.rglob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found under {root}")


@lru_cache(maxsize=None)
def load(name: str, usecols: tuple | None = None) -> pd.DataFrame:
    df = pd.read_csv(_path(name), usecols=list(usecols) if usecols else None, low_memory=False)
    for c in ("date", "transfer_date", "date_of_birth", "contract_expiration_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def season_of(d: pd.Series) -> pd.Series:
    """European season start year: Aug 2017 - Jun 2018 -> 2017. July counts as the new season."""
    d = pd.to_datetime(d)
    return (d.dt.year - (d.dt.month < 7).astype(int)).astype("Int64")
