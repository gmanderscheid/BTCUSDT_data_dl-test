"""
Téléchargement incrémental des données BTCUSDT depuis Binance Vision (data.binance.vision).

Datasets
--------
- spot_klines_1s      bougies 1 s du marché spot (fichiers klines 1s de Binance)
- futures_klines_1s   bougies 1 s du perpétuel USD-M, RECONSTRUITES à partir des aggTrades
                      (Binance ne publie pas de klines 1s pour les futures)
- futures_klines_1m   bougies 1 min officielles du perpétuel, pour vérifier les bougies
                      reconstruites (voir les tests)
- futures_funding     funding rate du perpétuel (toutes les 8 h)

Fonctionnement
--------------
- Récupère tous les mois depuis START (2020-01) jusqu'au mois dernier (fichiers mensuels),
  puis le mois en cours jour par jour (fichiers journaliers, jusqu'à hier).
- Stocke un fichier Parquet par mois : data/raw/<dataset>/<dataset>_<AAAA-MM>.parquet
- Ne rajoute jamais une ligne déjà présente : les nouvelles données sont fusionnées
  avec le fichier existant et dédoublonnées sur la clé temporelle.
- Un mois passé déjà présent sur disque n'est pas re-téléchargé (sauf --force).
- Les ZIP sont écrits sur disque (data/_downloads/) au fil du téléchargement, puis
  supprimés une fois traités : un fichier d'aggTrades de 700 Mo ne passe jamais en mémoire.
- Vérifie le SHA256 de chaque ZIP avec le fichier .CHECKSUM fourni par Binance.

Usage
-----
    python download_binance.py                                   # tous les datasets
    python download_binance.py --datasets futures_klines_1s      # un seul dataset
    python download_binance.py --datasets spot_klines_1s --force # re-télécharge tout
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

BASE = "https://data.binance.vision/data"
DATA_DIR = Path("data/raw")
DOWNLOAD_DIR = Path("data/_downloads")
SYMBOL = "BTCUSDT"
START = date(2020, 1, 1)

CHUNK_ROWS = 5_000_000   # lignes d'aggTrades lues à la fois (~ 1 Go de RAM au pic)
RETRIES = 3              # tentatives par fichier en cas d'erreur réseau

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
AGG_TRADE_COLS = [
    "agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker",
]

log = logging.getLogger("binance")
session = requests.Session()


# --------------------------------------------------------------------------- réseau

def download(rel_path: str, dest: Path) -> bool:
    """
    Télécharge un ZIP vers `dest` par blocs, en calculant son SHA256 au passage.
    Renvoie False si le fichier n'existe pas sur Binance (404).
    """
    url = f"{BASE}/{rel_path}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, RETRIES + 1):
        try:
            sha = hashlib.sha256()
            with session.get(url, stream=True, timeout=120) as r:
                if r.status_code == 404:
                    return False
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for block in r.iter_content(chunk_size=1 << 20):
                        f.write(block)
                        sha.update(block)

            chk = session.get(url + ".CHECKSUM", timeout=30)
            if chk.ok and chk.text.split()[0].strip().lower() != sha.hexdigest():
                raise ValueError(f"checksum invalide pour {url}")
            return True

        except (requests.RequestException, ValueError) as e:
            dest.unlink(missing_ok=True)
            if attempt == RETRIES:
                raise
            log.warning("%s : %s, nouvelle tentative (%d/%d)", rel_path, e, attempt + 1, RETRIES)
            time.sleep(5 * attempt)
    return False


# --------------------------------------------------------------------------- parsing commun

def to_datetime(s: pd.Series) -> pd.Series:
    """Convertit un timestamp Binance en datetime UTC (ms avant 2025, µs ensuite pour le spot)."""
    s = pd.to_numeric(s)
    unit = "us" if s.max() > 1e14 else "ms"
    return pd.to_datetime(s, unit=unit, utc=True)


def has_header(zip_path: Path) -> bool:
    """Certains CSV Binance (futures) commencent par une ligne d'en-tête, d'autres (spot) non."""
    with zipfile.ZipFile(zip_path) as z, z.open(z.namelist()[0]) as f:
        first = f.readline().decode().split(",")[0].strip()
    return not first.lstrip("-").isdigit()


def parse_table(zip_path: Path, spec: dict) -> pd.DataFrame:
    """Lecture d'un CSV « simple » (klines, funding) : une ligne du fichier = une ligne du résultat."""
    with zipfile.ZipFile(zip_path) as z, z.open(z.namelist()[0]) as f:
        df = pd.read_csv(f, header=None, skiprows=1 if has_header(zip_path) else 0)

    df = df.iloc[:, : len(spec["columns"])]
    df.columns = spec["columns"]
    df = df.drop(columns=["ignore"], errors="ignore")

    for col in df.columns:
        if col in spec["time_cols"]:
            df[col] = to_datetime(df[col])
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# --------------------------------------------------------------------------- aggTrades -> bougies 1 s

BAR_AGG = {
    "open": "first", "high": "max", "low": "min", "close": "last",
    "volume": "sum", "quote_volume": "sum", "n_trades": "sum",
    "taker_buy_base": "sum", "taker_buy_quote": "sum",
}


def trades_to_partial_bars(t: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège un morceau d'aggTrades en bougies 1 s.

    - open / close : prix du premier / dernier trade de la seconde (l'ordre du fichier est
      l'ordre chronologique d'exécution) ;
    - volume : somme des quantités en BTC ; quote_volume : somme de prix x quantité en USDT ;
    - n_trades : un aggTrade regroupe les trades first_trade_id..last_trade_id ;
    - taker_buy_* : trades où l'acheteur est le taker, c'est-à-dire is_buyer_maker == False.
    """
    price = pd.to_numeric(t["price"])
    qty = pd.to_numeric(t["quantity"])
    quote = price * qty
    taker_buy = ~t["is_buyer_maker"].astype(str).str.strip().str.lower().eq("true")

    frame = pd.DataFrame({
        "open_time": to_datetime(t["transact_time"]).dt.floor("s"),
        "open": price, "high": price, "low": price, "close": price,
        "volume": qty,
        "quote_volume": quote,
        "n_trades": pd.to_numeric(t["last_trade_id"]) - pd.to_numeric(t["first_trade_id"]) + 1,
        "taker_buy_base": qty.where(taker_buy, 0.0),
        "taker_buy_quote": quote.where(taker_buy, 0.0),
    })
    return frame.groupby("open_time", sort=True).agg(BAR_AGG)


def build_bars_from_aggtrades(zip_path: Path, spec: dict) -> pd.DataFrame:
    """
    Reconstruit les bougies 1 s d'un fichier d'aggTrades, en le lisant par morceaux.

    Une seconde peut être coupée entre deux morceaux : les bougies partielles sont donc
    ré-agrégées à la fin avec les mêmes règles (first / max / min / last / sum). Comme les
    morceaux sont lus dans l'ordre, « first » et « last » restent corrects.
    """
    skip = 1 if has_header(zip_path) else 0
    parts = []
    with zipfile.ZipFile(zip_path) as z, z.open(z.namelist()[0]) as f:
        reader = pd.read_csv(f, header=None, names=AGG_TRADE_COLS, skiprows=skip, chunksize=CHUNK_ROWS)
        for chunk in reader:
            parts.append(trades_to_partial_bars(chunk))

    bars = pd.concat(parts).groupby(level=0, sort=True).agg(BAR_AGG).reset_index()
    bars["close_time"] = bars["open_time"] + pd.Timedelta(milliseconds=999)
    bars["n_trades"] = bars["n_trades"].astype("int64")
    return bars[[c for c in KLINE_COLS if c != "ignore"]]


# --------------------------------------------------------------------------- datasets

def klines_spec(market: str, interval: str) -> dict:
    return {
        "monthly": f"{market}/monthly/klines/{SYMBOL}/{interval}/{SYMBOL}-{interval}-{{period}}.zip",
        "daily": f"{market}/daily/klines/{SYMBOL}/{interval}/{SYMBOL}-{interval}-{{period}}.zip",
        "columns": KLINE_COLS,
        "time_cols": ["open_time", "close_time"],
        "key": "open_time",
        "builder": parse_table,
    }


# Chaque dataset : chemins mensuel / journalier, colonnes, clé de dédoublonnage et fonction
# qui transforme un ZIP téléchargé en DataFrame.
DATASETS: dict[str, dict] = {
    "spot_klines_1s": klines_spec("spot", "1s"),
    "futures_klines_1s": {
        "monthly": f"futures/um/monthly/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-{{period}}.zip",
        "daily": f"futures/um/daily/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-{{period}}.zip",
        "columns": KLINE_COLS,
        "time_cols": ["open_time", "close_time"],
        "key": "open_time",
        "builder": build_bars_from_aggtrades,
    },
    "futures_klines_1m": klines_spec("futures/um", "1m"),
    "futures_funding": {
        "monthly": f"futures/um/monthly/fundingRate/{SYMBOL}/{SYMBOL}-fundingRate-{{period}}.zip",
        "daily": None,  # pas de fichiers journaliers pour le funding
        "columns": ["calc_time", "funding_interval_hours", "last_funding_rate"],
        "time_cols": ["calc_time"],
        "key": "calc_time",
        "builder": parse_table,
    },
}


# --------------------------------------------------------------------------- stockage

def merge_into(path: Path, new: pd.DataFrame, key: str) -> int:
    """Ajoute à `path` uniquement les lignes de `new` absentes du fichier. Renvoie le nb ajouté."""
    new = new.drop_duplicates(subset=key)
    if path.exists():
        old = pd.read_parquet(path)
        new = new[~new[key].isin(old[key])]
        if new.empty:
            return 0
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new

    combined = combined.sort_values(key).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(path)  # écriture atomique : pas de fichier corrompu si le script est interrompu
    return len(new)


def days_present(path: Path, key: str) -> set[date]:
    if not path.exists():
        return set()
    ts = pd.read_parquet(path, columns=[key])[key]
    return set(ts.dt.date.unique())


# --------------------------------------------------------------------------- boucle principale

def month_starts(start: date, end_exclusive: date):
    d = start.replace(day=1)
    while d < end_exclusive:
        yield d
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)


def fetch_and_store(rel_path: str, out: Path, spec: dict, label: str) -> None:
    """Télécharge un ZIP, le transforme avec le builder du dataset, fusionne, supprime le ZIP."""
    zip_path = DOWNLOAD_DIR / Path(rel_path).name
    try:
        if not download(rel_path, zip_path):
            log.info("%s : indisponible sur Binance Vision", label)
            return
        builder: Callable[[Path, dict], pd.DataFrame] = spec["builder"]
        added = merge_into(out, builder(zip_path, spec), spec["key"])
        log.info("%s : %d lignes ajoutées", label, added)
    finally:
        zip_path.unlink(missing_ok=True)


def sync_dataset(name: str, spec: dict, force: bool = False) -> None:
    today = date.today()
    current_month = today.replace(day=1)
    out_dir = DATA_DIR / name

    # 1) Mois complets : fichiers mensuels
    for m in month_starts(START, current_month):
        period = m.strftime("%Y-%m")
        out = out_dir / f"{name}_{period}.parquet"
        if out.exists() and not force:
            log.debug("%s %s déjà présent, ignoré", name, period)
            continue
        fetch_and_store(spec["monthly"].format(period=period), out, spec, f"{name} {period}")

    # 2) Mois en cours : fichiers journaliers jusqu'à hier
    if spec["daily"] is None:
        return
    out = out_dir / f"{name}_{current_month:%Y-%m}.parquet"
    already = days_present(out, spec["key"])
    d = current_month
    while d < today:
        if d not in already:
            fetch_and_store(spec["daily"].format(period=d.isoformat()), out, spec, f"{name} {d}")
        d += timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--force", action="store_true", help="re-télécharge les mois déjà présents")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    for name in args.datasets:
        sync_dataset(name, DATASETS[name], force=args.force)


if __name__ == "__main__":
    main()