"""
Détection et registre des trous (secondes manquantes) dans les klines 1s
(spot_klines_1s et futures_klines_1s).

Pourquoi ce module existe
-------------------------
Binance ne publie pas de bougie pour une seconde où il ne s'est échangé aucun bitcoin.
Les klines 1s ont donc des « trous » de deux natures très différentes :

- Trous COURTS (quelques secondes) : le marché était ouvert mais personne n'a traité
  pendant ce laps de temps. Le prix n'a pas bougé : on peut sans risque reporter le
  dernier prix et mettre le volume à 0.

- Trous LONGS (minutes, heures) : l'exchange était fermé (maintenance programmée, panne,
  incident technique). Le marché n'existait pas. Reporter le prix sur 3 heures ferait
  croire au modèle à un marché parfaitement immobile, ce qui n'arrive jamais. Il faut
  au contraire COUPER la série : aucune fenêtre de features ni aucune variable cible ne
  doit traverser ce trou, sinon le modèle apprend sur une réalité inventée.

Le seuil entre les deux est SHORT_GAP_MAX_SECONDS (60 s par défaut).

Le registre des trous longs
---------------------------
Tous les trous longs sont listés dans data/known_gaps.csv (colonnes : dataset, start,
end, duration_s, reason). Ce fichier a deux usages :

1. Pour les tests : un trou long présent dans le registre est considéré comme connu et
   accepté. Un trou long ABSENT du registre fait échouer les tests, car il peut s'agir
   d'un téléchargement raté plutôt que d'une vraie fermeture de l'exchange.
2. Pour la construction des jeux de données : c'est la liste des endroits où il faut
   couper les séries temporelles.

Usage
-----
    python gaps.py            # (re)construit le registre à partir des fichiers téléchargés
    python gaps.py --summary  # affiche seulement le résumé, sans écrire

La colonne `reason` est à remplir à la main (ex. « maintenance Binance ») : elle est
conservée quand on régénère le registre.
"""
from __future__ import annotations

import argparse
import os
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from download_binance import DATA_DIR

ROOT = Path(__file__).resolve().parent
# datasets à la seconde dont on suit les trous
SECOND_DATASETS = ["spot_klines_1s", "futures_klines_1s"]
SHORT_GAP_MAX_SECONDS = int(os.environ.get("SHORT_GAP_MAX_SECONDS", "60"))

FILE_RE = re.compile(r"_(\d{4})-(\d{2})\.parquet$")
GAP_COLUMNS = ["dataset", "start", "end", "duration_s", "reason"]
ONE_SECOND = pd.Timedelta(seconds=1)


def data_dir() -> Path:
    """Dossier des données brutes (surchargeable par BINANCE_DATA_DIR, utile pour les tests)."""
    return Path(os.environ.get("BINANCE_DATA_DIR", ROOT / DATA_DIR))


def known_gaps_path() -> Path:
    """Emplacement du registre : à côté du dossier raw/, sauf si KNOWN_GAPS_FILE est défini."""
    return Path(os.environ.get("KNOWN_GAPS_FILE", data_dir().parent / "known_gaps.csv"))


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
    - Premier fichier du dataset : Binance a pu commencer la publication en cours de mois,
      on démarre donc au premier jour réellement présent.
    - Mois en cours : les données s'arrêtent au dernier jour téléchargé (hier au mieux),
      on s'arrête donc à la fin de ce jour-là.
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


def split_gaps(gaps: pd.DataFrame, threshold: int = SHORT_GAP_MAX_SECONDS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les trous courts (<= threshold secondes) des trous longs."""
    long_mask = gaps["duration_s"] > threshold
    return gaps[~long_mask], gaps[long_mask]


# --------------------------------------------------------------------------- registre

def load_known_gaps(path: Path | None = None) -> pd.DataFrame:
    path = path or known_gaps_path()
    if not path.exists():
        return pd.DataFrame(columns=GAP_COLUMNS)
    df = pd.read_csv(path, dtype={"reason": "string"})
    df["start"] = pd.to_datetime(df["start"], utc=True)
    df["end"] = pd.to_datetime(df["end"], utc=True)
    df["reason"] = df["reason"].fillna("")
    return df


def scan_long_gaps(dataset: str) -> pd.DataFrame:
    """Parcourt tous les fichiers du dataset et renvoie l'ensemble des trous longs."""
    files = files_of(dataset)
    found = []
    for i, path in enumerate(files):
        df = pd.read_parquet(path, columns=["open_time"])
        start, end = expected_range(path, df, is_first_file=(i == 0))
        _, long_gaps = split_gaps(find_gaps(df["open_time"], start, end))
        found.append(long_gaps.assign(dataset=dataset))
    if not found:
        return pd.DataFrame(columns=GAP_COLUMNS)
    return pd.concat(found, ignore_index=True)


def build_registry(write: bool = True) -> pd.DataFrame:
    """Régénère le registre (tous les datasets 1 s) en conservant les `reason` saisies à la main."""
    scanned = pd.concat([scan_long_gaps(d) for d in SECOND_DATASETS], ignore_index=True)
    known = load_known_gaps()
    reasons = {(r.dataset, r.start, r.end): r.reason for r in known.itertuples()}
    scanned["reason"] = [reasons.get((r.dataset, r.start, r.end), "") for r in scanned.itertuples()]
    scanned = scanned[GAP_COLUMNS].sort_values(["dataset", "start"]).reset_index(drop=True)
    if write:
        path = known_gaps_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        scanned.to_csv(path, index=False)
    return scanned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", action="store_true", help="affiche sans écrire le registre")
    args = parser.parse_args()

    reg = build_registry(write=not args.summary)
    if reg.empty:
        print("Aucun trou long détecté.")
        return
    total_h = reg["duration_s"].sum() / 3600
    print(f"{len(reg)} trous longs (> {SHORT_GAP_MAX_SECONDS} s), {total_h:.1f} h au total")
    for dataset, g in reg.groupby("dataset"):
        print(f"  {dataset} : {len(g)} trous, {g['duration_s'].sum() / 3600:.1f} h")
    print()
    print(reg.sort_values("duration_s", ascending=False).head(20).to_string(index=False))
    if not args.summary:
        print(f"\nRegistre écrit dans {known_gaps_path()}")


if __name__ == "__main__":
    main()