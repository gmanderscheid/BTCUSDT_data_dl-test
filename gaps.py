"""
Détection des trous (secondes manquantes) dans les bougies 1 s, et registres d'exceptions.

Pourquoi ce module existe
-------------------------
Binance ne publie pas de bougie pour une seconde où il ne s'est échangé aucun bitcoin.
Les bougies 1 s ont donc des « trous » qu'il faut traiter différemment selon leur durée :

- Trous COURTS (<= SHORT_GAP_MAX_SECONDS, 60 s par défaut) : règle de traitement retenue,
  reporter le dernier prix et mettre le volume à 0.
- Trous LONGS : la série doit être COUPÉE. Aucune fenêtre de features ni aucune variable
  cible ne doit traverser le trou, sinon le modèle apprend sur une réalité inventée.

Le seuil de 60 s décrit une DURÉE : il ne démontre ni qu'un trou long est une maintenance,
ni qu'un trou court correspond à zéro transaction. C'est une règle de traitement, et la
cause de chaque trou long doit être examinée.

Les trous sont calculés fichier par fichier, puis FUSIONNÉS aux changements de mois : un
trou de 90 s coupé par minuit en 40 s + 50 s est bien un trou long (dataset_gaps).

Les registres : détection ≠ approbation
---------------------------------------
Deux registres versionnés recensent les exceptions acceptées :

- data/known_gaps.csv (trous longs), rempli par `python gaps.py` ;
- data/known_minute_mismatches.csv (minutes du perpétuel irréparables ou non fiables),
  rempli par `python diagnose_perp.py --register`.

Chaque entrée a un statut :
- "candidate" : détectée automatiquement, PAS ENCORE EXAMINÉE. Les tests échouent tant
  qu'une entrée reste candidate : l'inscription ne vaut pas acceptation.
- "approved"  : examinée et acceptée, avec une raison obligatoire (colonne `reason`).

L'approbation se fait avec `python approve.py` (ou en éditant le CSV). Les tests
échouent aussi sur une entrée approuvée qui ne correspond plus aux données (exception
obsolète). Les registres servent enfin à construire le masque des features : ce sont
les endroits où couper les séries.

Usage
-----
    python gaps.py            # (re)construit le registre des trous longs
    python gaps.py --summary  # affiche seulement le résumé, sans écrire
"""
from __future__ import annotations

import argparse
import os
import re
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd

from download_binance import DATA_DIR

ROOT = Path(__file__).resolve().parent
# datasets à la seconde dont on suit les trous
SECOND_DATASETS = ["spot_klines_1s", "futures_klines_1s"]
SHORT_GAP_MAX_SECONDS = int(os.environ.get("SHORT_GAP_MAX_SECONDS", "60"))

FILE_RE = re.compile(r"_(\d{4})-(\d{2})\.parquet$")
ONE_SECOND = pd.Timedelta(seconds=1)
STATUSES = ("candidate", "approved")

GAP_COLUMNS = ["dataset", "start", "end", "duration_s", "status", "reason"]
MISMATCH_SIDES = ("absente_chez_nous", "absente_chez_binance", "zone_invalide")
MISMATCH_COLUMNS = ["side", "minute", "official_volume", "status", "reason"]


def data_dir() -> Path:
    """Dossier des données brutes (surchargeable par BINANCE_DATA_DIR, utile pour les tests)."""
    return Path(os.environ.get("BINANCE_DATA_DIR", ROOT / DATA_DIR))


def known_gaps_path() -> Path:
    """Registre des trous longs : à côté du dossier raw/, sauf si KNOWN_GAPS_FILE est défini."""
    return Path(os.environ.get("KNOWN_GAPS_FILE", data_dir().parent / "known_gaps.csv"))


def known_mismatches_path() -> Path:
    """
    Registre des minutes du perpétuel qui diffèrent de façon irréparable des bougies 1 min
    officielles (data/known_minute_mismatches.csv).

    side = "absente_chez_nous"    : Binance a une bougie officielle, mais aucun aggTrade,
                                    même dans le fichier journalier ;
    side = "absente_chez_binance" : nous avons des aggTrades, mais le fichier officiel 1 min
                                    n'a pas de bougie, même dans le fichier journalier ;
    side = "zone_invalide"        : minute ajoutée À LA MAIN, présente des deux côtés mais
                                    jugée non fiable (incident chez Binance). Exclue des
                                    comparaisons et de la construction des features.
    """
    return Path(os.environ.get("KNOWN_MISMATCHES_FILE", data_dir().parent / "known_minute_mismatches.csv"))


# --------------------------------------------------------------------------- dates

def files_of(dataset: str, root: Path | None = None) -> list[Path]:
    root = root or data_dir()
    return sorted((root / dataset).glob(f"{dataset}_*.parquet"))


def month_of(path: Path) -> date:
    """Mois couvert par un fichier, déduit de son nom (…_AAAA-MM.parquet)."""
    y, m = FILE_RE.search(path.name).groups()
    return date(int(y), int(m), 1)


def next_month(d: date) -> date:
    return (d.replace(day=28) + timedelta(days=4)).replace(day=1)


def utc(d: date) -> pd.Timestamp:
    return pd.Timestamp(d, tz="UTC")


def expected_range(path: Path, df: pd.DataFrame, is_first_file: bool,
                   today: date | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Intervalle [start, end[ dans lequel on attend une bougie par seconde.

    - Cas général : le mois complet.
    - Premier fichier du dataset : on démarre au premier jour réellement présent. Le
      contrat de couverture (data_contract.py) vérifie séparément que ce premier mois est
      bien celui attendu.
    - Mois en cours : on s'arrête à la fin du dernier jour téléchargé.
    """
    today = today or date.today()
    month = month_of(path)
    start, end = utc(month), utc(next_month(month))
    if is_first_file:
        start = max(start, df["open_time"].min().floor("D"))
    if month == today.replace(day=1):
        end = min(end, df["open_time"].max().floor("D") + pd.Timedelta(days=1))
    return start, end


# --------------------------------------------------------------------------- détection

def find_gaps(timestamps: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Liste toutes les plages de secondes manquantes dans [start, end[.

    Renvoie un DataFrame (start, end, duration_s) où `start` et `end` sont la première
    et la dernière seconde MANQUANTE de chaque trou (bornes incluses).

    Calcul en O(n) : on compare chaque timestamp au suivant, sans construire la grille
    complète du mois (2,6 millions de secondes).
    """
    ts = pd.DatetimeIndex(timestamps).unique().sort_values()
    ts = ts[(ts >= start) & (ts < end)]
    # sentinelles : une seconde « présente » juste avant start et exactement à end
    bounds = pd.DatetimeIndex([start - ONE_SECOND]).append(ts).append(pd.DatetimeIndex([end]))
    diffs = bounds[1:] - bounds[:-1]
    idx = (diffs > ONE_SECOND).nonzero()[0]
    gaps = pd.DataFrame({
        "start": bounds[idx] + ONE_SECOND,
        "end": bounds[idx + 1] - ONE_SECOND,
        "duration_s": (diffs[idx] // ONE_SECOND - 1).astype("int64"),
    })
    return gaps.reset_index(drop=True)


def merge_adjacent_gaps(gaps: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne les trous qui se touchent (fin + 1 s == début du suivant).

    Cas typique : un trou coupé par un changement de mois, qui apparaît comme deux trous
    distincts (fin du fichier du mois M, début du fichier du mois M+1).
    """
    if gaps.empty:
        return gaps
    g = gaps.sort_values("start").reset_index(drop=True)
    new_group = g["start"] != g["end"].shift() + ONE_SECOND
    grp = new_group.cumsum()
    out = g.groupby(grp).agg(start=("start", "first"), end=("end", "last"))
    out["duration_s"] = ((out["end"] - out["start"]) // ONE_SECOND + 1).astype("int64")
    return out.reset_index(drop=True)


@lru_cache(maxsize=None)
def dataset_gaps(dataset: str, root: str | None = None, today: date | None = None) -> pd.DataFrame:
    """
    Tous les trous d'un dataset 1 s, fusionnés aux changements de mois.

    Ne lit que la colonne open_time de chaque fichier. Mis en cache : les tests appellent
    cette fonction une fois par dataset, puis filtrent par fichier.
    """
    files = files_of(dataset, Path(root) if root else None)
    parts = []
    for i, path in enumerate(files):
        df = pd.read_parquet(path, columns=["open_time"])
        start, end = expected_range(path, df, is_first_file=(i == 0), today=today)
        parts.append(find_gaps(df["open_time"], start, end))
    if not parts:
        return pd.DataFrame(columns=["start", "end", "duration_s"])
    return merge_adjacent_gaps(pd.concat(parts, ignore_index=True))


def split_gaps(gaps: pd.DataFrame, threshold: int = SHORT_GAP_MAX_SECONDS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les trous courts (<= threshold secondes) des trous longs."""
    long_mask = gaps["duration_s"] > threshold
    return gaps[~long_mask], gaps[long_mask]


def gaps_overlapping(gaps: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Trous qui chevauchent [start, end[ (un trou à cheval sur deux mois apparaît dans les deux)."""
    return gaps[(gaps["start"] < end) & (gaps["end"] >= start)]


# --------------------------------------------------------------------------- registres

def _normalize_review(df: pd.DataFrame) -> pd.DataFrame:
    """Statut et raison : une entrée sans statut (ancien format) est une candidate."""
    if "status" not in df.columns:
        df["status"] = "candidate"
    df["status"] = df["status"].fillna("candidate").astype(str).str.strip()
    if "reason" not in df.columns:
        df["reason"] = ""
    df["reason"] = df["reason"].fillna("").astype(str)
    return df


def load_known_gaps(path: Path | None = None) -> pd.DataFrame:
    path = path or known_gaps_path()
    if not path.exists():
        return pd.DataFrame(columns=GAP_COLUMNS)
    df = pd.read_csv(path, dtype={"reason": "string", "status": "string"})
    df["start"] = pd.to_datetime(df["start"], utc=True)
    df["end"] = pd.to_datetime(df["end"], utc=True)
    return _normalize_review(df)[GAP_COLUMNS]


def load_known_mismatches(path: Path | None = None) -> pd.DataFrame:
    path = path or known_mismatches_path()
    if not path.exists():
        return pd.DataFrame(columns=MISMATCH_COLUMNS)
    df = pd.read_csv(path, dtype={"reason": "string", "status": "string"})
    df["minute"] = pd.to_datetime(df["minute"], utc=True)
    return _normalize_review(df)[MISMATCH_COLUMNS]


def approved(df: pd.DataFrame) -> pd.DataFrame:
    """Entrées examinées et acceptées, avec une raison."""
    return df[(df["status"] == "approved") & (df["reason"].str.strip() != "")]


def detected_long_gaps() -> pd.DataFrame:
    parts = [split_gaps(dataset_gaps(d, str(data_dir())))[1].assign(dataset=d) for d in SECOND_DATASETS]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["dataset", "start", "end", "duration_s"])
    return pd.concat(parts, ignore_index=True)


def build_registry(write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Régénère le registre des trous longs.

    Un trou déjà présent garde son statut et sa raison. Un nouveau trou est inscrit comme
    « candidate » : il faudra l'examiner puis l'approuver. Les entrées qui ne correspondent
    plus à aucun trou détecté sont retirées et renvoyées pour affichage.
    """
    dataset_gaps.cache_clear()
    scanned = detected_long_gaps()
    known = load_known_gaps()
    review = {(r.dataset, r.start, r.end): (r.status, r.reason) for r in known.itertuples()}
    scanned["status"] = [review.get((r.dataset, r.start, r.end), ("candidate", ""))[0] for r in scanned.itertuples()]
    scanned["reason"] = [review.get((r.dataset, r.start, r.end), ("candidate", ""))[1] for r in scanned.itertuples()]
    scanned = scanned[GAP_COLUMNS].sort_values(["dataset", "start"]).reset_index(drop=True)

    keys = set(zip(scanned["dataset"], scanned["start"], scanned["end"]))
    removed = known[[(r.dataset, r.start, r.end) not in keys for r in known.itertuples()]]
    if write:
        path = known_gaps_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        scanned.to_csv(path, index=False)
    return scanned, removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", action="store_true", help="affiche sans écrire le registre")
    args = parser.parse_args()

    reg, removed = build_registry(write=not args.summary)
    if reg.empty:
        print("Aucun trou long détecté.")
    else:
        total_h = reg["duration_s"].sum() / 3600
        print(f"{len(reg)} trous longs (> {SHORT_GAP_MAX_SECONDS} s), {total_h:.1f} h au total")
        for dataset, g in reg.groupby("dataset"):
            print(f"  {dataset} : {len(g)} trous, {g['duration_s'].sum() / 3600:.1f} h")
        cand = reg[reg["status"] != "approved"]
        print(f"\n{len(cand)} trou(s) à examiner (statut candidate) :")
        if not cand.empty:
            print(cand.sort_values("duration_s", ascending=False).head(30).to_string(index=False))
            print("\nUne journée entière (86 400 s) n'est jamais une maintenance : vérifie d'abord "
                  "le téléchargement. Approuve ensuite avec : python approve.py gaps <date> \"<raison>\"")
    if not removed.empty:
        print(f"\n{len(removed)} entrée(s) retirée(s) du registre car plus détectée(s) :")
        print(removed.to_string(index=False))
    if not args.summary:
        print(f"\nRegistre écrit dans {known_gaps_path()}")


if __name__ == "__main__":
    main()