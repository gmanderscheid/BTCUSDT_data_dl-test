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
- Un mois passé déjà présent sur disque n'est pas re-téléchargé. Avec --force, il est
  re-téléchargé et REMPLACE entièrement l'ancien contenu (utile après une correction du
  code de reconstruction : une simple fusion garderait les anciennes valeurs).
- Réparation des jours manquants : les fichiers mensuels de Binance Vision omettent parfois
  des journées entières. Pour chaque mois, les jours totalement absents sont recherchés dans
  les fichiers journaliers et ajoutés. Pour le perpétuel, les journées PARTIELLEMENT
  manquantes sont aussi détectées en croisant les bougies 1 s et les bougies 1 min
  officielles, ainsi que les journées dont le VOLUME diffère de la référence alors que
  toutes les minutes sont présentes (désactivable avec --no-repair). Une journée n'est
  remplacée par son fichier journalier que si celui-ci couvre au moins les mêmes instants.
  Chaque tentative est notée dans data/repair_attempts.csv et n'est pas refaite aux
  lancements suivants (sauf --retry-repairs).
- Les ZIP sont écrits sur disque (data/_downloads/) au fil du téléchargement, puis
  supprimés une fois traités : un fichier d'aggTrades de 700 Mo ne passe jamais en mémoire.
- Vérifie OBLIGATOIREMENT le SHA256 de chaque ZIP avec le fichier .CHECKSUM de Binance :
  un checksum faux, vide, mal formé ou indisponible fait échouer le téléchargement
  (--allow-missing-checksum accepte explicitement l'absence de .CHECKSUM, avec avertissement).
- Trace la provenance de chaque fichier intégré dans data/manifest.csv : source, SHA256,
  date de récupération, mode (fusion / remplacement) et empreinte du code de transformation.

Usage
-----
    python download_binance.py                                   # tous les datasets
    python download_binance.py --datasets futures_klines_1s      # un seul dataset
    python download_binance.py --datasets spot_klines_1s --force # re-télécharge et remplace tout
    python download_binance.py --no-repair                       # sans réparation des jours manquants
    python download_binance.py --retry-repairs                   # refait les réparations déjà tentées
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import re
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests

BASE = "https://data.binance.vision/data"
DATA_DIR = Path("data/raw")
DOWNLOAD_DIR = Path("data/_downloads")
MANIFEST = Path("data/manifest.csv")
REPAIR_LOG = Path("data/repair_attempts.csv")
SYMBOL = "BTCUSDT"
START = date(2020, 1, 1)

CHUNK_ROWS = 5_000_000   # lignes d'aggTrades lues à la fois (~ 1 Go de RAM au pic)
RETRIES = 3              # tentatives par fichier en cas d'erreur réseau
ALLOW_MISSING_CHECKSUM = False   # --allow-missing-checksum
DAY_VOLUME_REPAIR_THRESHOLD = 0.005  # écart de volume journalier (vs 1 min officiel) qui déclenche une réparation

# empreinte du code de transformation, enregistrée dans le manifeste
CODE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]

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

class ChecksumError(Exception):
    """Le SHA256 d'un ZIP n'a pas pu être vérifié (faux, vide, mal formé ou indisponible)."""


class ChecksumUnavailable(ChecksumError):
    """Binance ne publie pas de .CHECKSUM pour ce fichier (réponse 404)."""


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def fetch_checksum(url: str) -> str | None:
    """
    Renvoie le SHA256 attendu, lu dans le fichier .CHECKSUM (format « <sha256>  <nom> »).
    None si Binance ne publie pas de .CHECKSUM (404). Toute autre anomalie lève une erreur.
    """
    chk = session.get(url + ".CHECKSUM", timeout=30)
    if chk.status_code == 404:
        return None
    chk.raise_for_status()                      # 5xx, 403... : erreur réseau, retentée
    parts = chk.text.split()
    if not parts or not SHA256_RE.match(parts[0].strip().lower()):
        raise ChecksumError(f"fichier .CHECKSUM vide ou mal formé pour {url}")
    return parts[0].strip().lower()


def download(rel_path: str, dest: Path) -> str | None:
    """
    Télécharge un ZIP vers `dest` par blocs et vérifie OBLIGATOIREMENT son SHA256.

    Renvoie le SHA256 du fichier, ou None si le ZIP n'existe pas sur Binance (404).
    Lève ChecksumError si l'empreinte est fausse ou illisible, et ChecksumUnavailable si
    Binance ne publie pas de .CHECKSUM (sauf ALLOW_MISSING_CHECKSUM). En cas d'échec, le
    fichier partiel est supprimé. Les erreurs réseau et les empreintes fausses sont
    retentées RETRIES fois.
    """
    url = f"{BASE}/{rel_path}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, RETRIES + 1):
        try:
            sha = hashlib.sha256()
            with session.get(url, stream=True, timeout=120) as r:
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for block in r.iter_content(chunk_size=1 << 20):
                        f.write(block)
                        sha.update(block)

            expected = fetch_checksum(url)
            if expected is None:
                if not ALLOW_MISSING_CHECKSUM:
                    raise ChecksumUnavailable(
                        f"pas de .CHECKSUM pour {url} : relance avec --allow-missing-checksum "
                        "pour accepter ce fichier sans vérification")
                log.warning("%s : pas de .CHECKSUM, fichier accepté SANS vérification", rel_path)
            elif expected != sha.hexdigest():
                raise ChecksumError(f"checksum invalide pour {url}")
            return sha.hexdigest()

        except ChecksumUnavailable:
            dest.unlink(missing_ok=True)
            raise
        except (requests.RequestException, ChecksumError) as e:
            dest.unlink(missing_ok=True)
            if attempt == RETRIES:
                raise
            log.warning("%s : %s, nouvelle tentative (%d/%d)", rel_path, e, attempt + 1, RETRIES)
            time.sleep(5 * attempt)
    return None


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


class AggTradeOrderError(Exception):
    """Les identifiants d'aggTrades ne sont pas croissants : l'hypothèse d'ordre est violée."""


class IdRanges:
    """
    Ensemble compact des identifiants déjà vus, stocké sous forme d'intervalles [début, fin].

    Les agg_trade_id d'un fichier sont (presque) consécutifs : quelques intervalles suffisent
    à représenter des dizaines de millions d'identifiants, là où un set Python coûterait
    plusieurs Go.
    """

    def __init__(self) -> None:
        self.starts = np.empty(0, dtype=np.int64)
        self.ends = np.empty(0, dtype=np.int64)

    def contains(self, ids: np.ndarray) -> np.ndarray:
        if not len(self.starts):
            return np.zeros(len(ids), dtype=bool)
        pos = np.searchsorted(self.starts, ids, side="right") - 1
        ok = pos >= 0
        res = np.zeros(len(ids), dtype=bool)
        res[ok] = ids[ok] <= self.ends[pos[ok]]
        return res

    def add(self, ids: np.ndarray) -> None:
        """Ajoute des identifiants triés, uniques."""
        if not len(ids):
            return
        breaks = np.flatnonzero(np.diff(ids) != 1)
        starts = np.concatenate([ids[:1], ids[breaks + 1]])
        ends = np.concatenate([ids[breaks], ids[-1:]])
        s = np.concatenate([self.starts, starts])
        e = np.concatenate([self.ends, ends])
        order = np.argsort(s, kind="stable")
        s, e = s[order], e[order]
        # fusion des intervalles contigus ou chevauchants
        merged_s, merged_e = [s[0]], [e[0]]
        for a, b in zip(s[1:], e[1:]):
            if a <= merged_e[-1] + 1:
                merged_e[-1] = max(merged_e[-1], b)
            else:
                merged_s.append(a)
                merged_e.append(b)
        self.starts = np.array(merged_s, dtype=np.int64)
        self.ends = np.array(merged_e, dtype=np.int64)


def build_bars_from_aggtrades(zip_path: Path, spec: dict) -> pd.DataFrame:
    """
    Reconstruit les bougies 1 s d'un fichier d'aggTrades, en le lisant par morceaux.

    Une seconde peut être coupée entre deux morceaux : les bougies partielles sont donc
    ré-agrégées à la fin avec les mêmes règles (first / max / min / last / sum). Comme les
    morceaux sont lus dans l'ordre, « first » et « last » restent corrects.

    Hypothèse d'ordre, VÉRIFIÉE : open et close reposent sur l'ordre du fichier. Une fois
    les doublons retirés, les agg_trade_id doivent être strictement croissants. Un
    identifiant jamais vu mais inférieur au plus grand déjà lu viole cette hypothèse et
    lève AggTradeOrderError, au lieu d'être supprimé silencieusement.

    Dédoublonnage : certains fichiers de Binance Vision contiennent des aggTrades en double
    (constaté les 12 et 13/09/2022, où le volume était exactement doublé). Un identifiant
    déjà vu, dans le même morceau ou dans un morceau précédent, est ignoré. Les
    identifiants vus sont conservés sous forme d'intervalles (IdRanges).
    """
    skip = 1 if has_header(zip_path) else 0
    parts = []
    seen = IdRanges()
    max_seen = -1
    dropped = 0
    last_time = None
    time_disorder = 0
    with zipfile.ZipFile(zip_path) as z, z.open(z.namelist()[0]) as f:
        reader = pd.read_csv(f, header=None, names=AGG_TRADE_COLS, skiprows=skip, chunksize=CHUNK_ROWS)
        for chunk in reader:
            ids = pd.to_numeric(chunk["agg_trade_id"]).to_numpy(dtype=np.int64)
            dup = pd.Series(ids).duplicated().to_numpy() | seen.contains(ids)
            dropped += int(dup.sum())
            new_ids = ids[~dup]
            if len(new_ids):
                sequence = np.concatenate([[max_seen], new_ids])
                if not (np.diff(sequence) > 0).all():
                    bad = new_ids[np.flatnonzero(np.diff(sequence) <= 0)[0]]
                    raise AggTradeOrderError(
                        f"{zip_path.name} : identifiant d'aggTrade {bad} inédit mais hors ordre "
                        f"(plus grand identifiant déjà lu : {max(max_seen, int(new_ids.max()))}). "
                        "L'ordre du fichier ne peut pas servir à déterminer open / close.")
                max_seen = int(new_ids[-1])
                seen.add(new_ids)
            chunk = chunk[~dup]
            if chunk.empty:
                continue
            times = pd.to_numeric(chunk["transact_time"]).to_numpy()
            seq_t = times if last_time is None else np.concatenate([[last_time], times])
            time_disorder += int((np.diff(seq_t) < 0).sum())
            last_time = times[-1]
            parts.append(trades_to_partial_bars(chunk))
    if dropped:
        log.warning("%s : %d aggTrades en double ignorés", zip_path.name, dropped)
    if time_disorder:
        log.warning("%s : %d horodatages en recul malgré des identifiants croissants", zip_path.name, time_disorder)

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

    # écriture atomique : pas de fichier corrompu si le script est interrompu
    write_atomic(path, combined.sort_values(key).reset_index(drop=True))
    return len(new)


def write_atomic(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def replace_range(path: Path, new: pd.DataFrame, key: str,
                  start: pd.Timestamp, end: pd.Timestamp) -> int:
    """
    Remplace, dans `path`, toutes les lignes dont la clé est dans [start, end[ par `new`.

    Pourquoi : merge_into n'ajoute que des lignes absentes. Après une correction du code de
    reconstruction, ou pour compléter une seconde partiellement reconstruite, il faut au
    contraire REMPLACER les anciennes valeurs. On remplace une plage entière, sans jamais
    additionner deux bougies de la même seconde (ce qui recompterait les mêmes trades).
    L'écriture est atomique. Renvoie le nombre de lignes écrites dans la plage.
    """
    new = new.drop_duplicates(subset=key)
    new = new[(new[key] >= start) & (new[key] < end)]
    if path.exists():
        old = pd.read_parquet(path)
        old = old[(old[key] < start) | (old[key] >= end)]
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    write_atomic(path, combined.sort_values(key).reset_index(drop=True))
    return len(new)


def record_manifest(dataset: str, target: Path, source: str, sha: str, mode: str, rows: int) -> None:
    """Ajoute une ligne de provenance dans data/manifest.csv."""
    row = pd.DataFrame([{
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset, "target": target.name, "source": f"{BASE}/{source}",
        "sha256": sha, "mode": mode, "rows": rows, "code_sha": CODE_SHA,
    }])
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(MANIFEST, mode="a", header=not MANIFEST.exists(), index=False)


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


def dataset_of(out: Path) -> str:
    return out.parent.name


def fetch_and_store(rel_path: str, out: Path, spec: dict, label: str, mode: str = "merge",
                    period: tuple[pd.Timestamp, pd.Timestamp] | None = None) -> pd.DataFrame | None:
    """
    Télécharge un ZIP, le transforme avec le builder du dataset, l'intègre, supprime le ZIP.

    mode = "merge"   : ajoute les lignes absentes (comportement par défaut) ;
    mode = "replace" : remplace toutes les lignes de `period` par le nouveau contenu ;
    mode = "replace_if_superset" : remplace `period` seulement si le nouveau contenu couvre
                       au moins tous les instants déjà présents, sinon fusionne.
    Renvoie le DataFrame obtenu (None si le fichier n'existe pas chez Binance).
    """
    zip_path = DOWNLOAD_DIR / Path(rel_path).name
    try:
        sha = download(rel_path, zip_path)
        if sha is None:
            log.info("%s : indisponible sur Binance Vision", label)
            return None
        builder: Callable[[Path, dict], pd.DataFrame] = spec["builder"]
        new = builder(zip_path, spec)
        key = spec["key"]

        if mode == "replace_if_superset" and out.exists():
            old_keys = pd.read_parquet(out, columns=[key])[key]
            old_keys = old_keys[(old_keys >= period[0]) & (old_keys < period[1])]
            mode = "replace" if old_keys.isin(new[key]).all() else "merge"
            if mode == "merge":
                log.warning("%s : le fichier journalier ne couvre pas tout l'existant, simple fusion", label)
        elif mode == "replace_if_superset":
            mode = "replace"

        if mode == "replace":
            rows = replace_range(out, new, key, *period)
            log.info("%s : %d lignes (plage remplacée)", label, rows)
        else:
            rows = merge_into(out, new, key)
            log.info("%s : %d lignes ajoutées", label, rows)
        record_manifest(dataset_of(out), out, rel_path, sha, mode, rows)
        return new
    finally:
        zip_path.unlink(missing_ok=True)


def month_bounds(m: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(m, tz="UTC")
    return start, start + pd.offsets.MonthBegin(1)


def day_bounds(d: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(d, tz="UTC")
    return start, start + pd.Timedelta(days=1)


def fill_missing_days(name: str, spec: dict, out: Path, first_day: date, end_day: date) -> None:
    """
    Télécharge les fichiers journaliers des jours de [first_day, end_day[ totalement absents de `out`.

    Pourquoi : les fichiers mensuels de Binance Vision omettent parfois des journées entières
    (constaté sur les aggTrades du perpétuel). Sans réparation, ces journées apparaissent comme
    des « trous longs » que l'on pourrait prendre à tort pour des fermetures de l'exchange.
    Si le fichier journalier n'existe pas non plus, le jour reste manquant et les tests le signaleront.
    """
    already = days_present(out, spec["key"])
    d = first_day
    while d < end_day:
        if d not in already:
            fetch_and_store(spec["daily"].format(period=d.isoformat()), out, spec, f"{name} {d}",
                            mode="replace", period=day_bounds(d))
        d += timedelta(days=1)


def sync_dataset(name: str, spec: dict, force: bool = False, repair: bool = True) -> None:
    today = date.today()
    current_month = today.replace(day=1)
    out_dir = DATA_DIR / name

    # 1) Mois complets : fichiers mensuels, puis réparation des jours manquants
    for m in month_starts(START, current_month):
        period = m.strftime("%Y-%m")
        out = out_dir / f"{name}_{period}.parquet"
        if out.exists() and not force:
            log.debug("%s %s déjà présent, ignoré", name, period)
        else:
            # --force : le nouveau contenu REMPLACE le mois entier
            fetch_and_store(spec["monthly"].format(period=period), out, spec, f"{name} {period}",
                            mode="replace" if force else "merge", period=month_bounds(m))

        # on ne répare que les mois dont Binance a publié au moins une partie
        if repair and spec["daily"] is not None and out.exists():
            month_end = (m.replace(day=28) + timedelta(days=4)).replace(day=1)
            fill_missing_days(name, spec, out, m, month_end)

    # 2) Mois en cours : fichiers journaliers jusqu'à hier
    if spec["daily"] is None:
        return
    out = out_dir / f"{name}_{current_month:%Y-%m}.parquet"
    fill_missing_days(name, spec, out, current_month, today)


def load_repair_log() -> set[tuple[str, str]]:
    if not REPAIR_LOG.exists():
        return set()
    df = pd.read_csv(REPAIR_LOG, dtype=str)
    return set(zip(df["dataset"], df["day"]))


def log_repair(dataset: str, d: date, reason: str, rows: int | None) -> None:
    row = pd.DataFrame([{
        "attempted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset, "day": d.isoformat(), "reason": reason,
        "result": "indisponible" if rows is None else f"{rows} lignes",
    }])
    REPAIR_LOG.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(REPAIR_LOG, mode="a", header=not REPAIR_LOG.exists(), index=False)


def perp_day_discrepancies(path_1s: Path, path_1m: Path) -> dict[str, dict[date, str]]:
    """
    Journées du perpétuel à réparer, par dataset, avec la raison :
    - minutes officielles absentes chez nous       -> futures_klines_1s
    - minutes à nous absentes des bougies officielles -> futures_klines_1m
    - volume journalier différent de plus de DAY_VOLUME_REPAIR_THRESHOLD alors que les
      minutes sont présentes (journée amputée en contenu) -> les deux datasets
    """
    s = pd.read_parquet(path_1s, columns=["open_time", "volume"])
    m = pd.read_parquet(path_1m, columns=["open_time", "volume"])
    m = m[m["volume"] > 0]
    ours = pd.DatetimeIndex(s["open_time"].dt.floor("min").unique())
    off = pd.DatetimeIndex(m["open_time"].unique())

    todo: dict[str, dict[date, str]] = {"futures_klines_1s": {}, "futures_klines_1m": {}}
    for d in set(off.difference(ours).date):
        todo["futures_klines_1s"][d] = "minutes officielles absentes chez nous"
    for d in set(ours.difference(off).date):
        todo["futures_klines_1m"][d] = "minutes absentes des bougies officielles"

    vs = s.groupby(s["open_time"].dt.date)["volume"].sum()
    vm = m.groupby(m["open_time"].dt.date)["volume"].sum()
    common = vs.index.intersection(vm.index)
    rel = (vs[common] / vm[common] - 1).abs()
    for d in rel[rel > DAY_VOLUME_REPAIR_THRESHOLD].index:
        for ds in todo:
            todo[ds].setdefault(d, f"volume journalier différent de {rel[d]:.2%}")
    return todo


def repair_perp_partial_days(retry: bool = False) -> None:
    """
    Répare les journées du perpétuel incomplètes, en croisant les bougies 1 s reconstruites
    et les bougies 1 min officielles.

    Pourquoi : certains fichiers mensuels de Binance Vision sont tronqués au milieu d'une
    journée (par exemple le 13/04/2020 à partir de 00:32), ou amputés en contenu sans qu'une
    minute disparaisse. fill_missing_days ne voit que les journées totalement vides. Ici,
    chaque journée signalée par perp_day_discrepancies est re-téléchargée depuis son fichier
    journalier. Le contenu du jour est REMPLACÉ si le fichier journalier couvre au moins les
    mêmes instants (sinon simple fusion), pour compléter aussi les secondes partielles.

    Chaque tentative est notée dans data/repair_attempts.csv et n'est pas refaite aux
    lancements suivants, sauf retry=True. Si le fichier journalier est lui aussi
    incomplet, les tests de vérification continueront de signaler la journée.
    """
    attempted = set() if retry else load_repair_log()
    dirs = {ds: DATA_DIR / ds for ds in ("futures_klines_1s", "futures_klines_1m")}
    for path_1s in sorted(dirs["futures_klines_1s"].glob("futures_klines_1s_*.parquet")):
        period = path_1s.stem.rsplit("_", 1)[-1]
        path_1m = dirs["futures_klines_1m"] / f"futures_klines_1m_{period}.parquet"
        if not path_1m.exists():
            continue
        targets = {"futures_klines_1s": path_1s, "futures_klines_1m": path_1m}
        for ds, days in perp_day_discrepancies(path_1s, path_1m).items():
            spec = DATASETS[ds]
            for d, reason in sorted(days.items()):
                if (ds, d.isoformat()) in attempted:
                    log.debug("%s %s : réparation déjà tentée, ignorée", ds, d)
                    continue
                log.info("%s %s : %s, nouvelle tentative avec le fichier journalier", ds, d, reason)
                new = fetch_and_store(spec["daily"].format(period=d.isoformat()), targets[ds], spec,
                                      f"{ds} {d}", mode="replace_if_superset", period=day_bounds(d))
                log_repair(ds, d, reason, None if new is None else len(new))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--force", action="store_true", help="re-télécharge les mois déjà présents")
    parser.add_argument("--no-repair", action="store_true",
                        help="ne cherche pas les jours manquants dans les fichiers journaliers")
    parser.add_argument("--retry-repairs", action="store_true",
                        help="refait les réparations déjà tentées (data/repair_attempts.csv)")
    parser.add_argument("--allow-missing-checksum", action="store_true",
                        help="accepte, avec avertissement, les fichiers sans .CHECKSUM chez Binance")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    global ALLOW_MISSING_CHECKSUM
    ALLOW_MISSING_CHECKSUM = args.allow_missing_checksum

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    for name in args.datasets:
        sync_dataset(name, DATASETS[name], force=args.force, repair=not args.no_repair)

    # réparation croisée des journées partiellement manquantes du perpétuel
    if not args.no_repair and {"futures_klines_1s", "futures_klines_1m"} & set(args.datasets):
        repair_perp_partial_days(retry=args.retry_repairs)


if __name__ == "__main__":
    main()