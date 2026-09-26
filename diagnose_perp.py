"""
Diagnostic du perpétuel : compare, mois par mois, nos bougies 1 s reconstruites (agrégées
en 1 min) aux bougies 1 min officielles de Binance.

Sert à comprendre les échecs de `pytest -m verification` et à régler ses seuils
(MAX_VOLUME_DEFICIT, MAX_VOLUME_DRIFT) sur des chiffres réels.

Usage :
    python diagnose_perp.py                 # tous les mois
    python diagnose_perp.py 2021-03 2024-10 # seulement ces mois
    python diagnose_perp.py --register      # enregistre les minutes manquantes restantes comme
                                            # irréparables (data/known_minute_mismatches.csv)

À n'utiliser qu'APRÈS download_binance.py, qui tente de réparer ces minutes avec les
fichiers journaliers : --register ne doit enregistrer que ce que Binance ne fournit pas.
La colonne `reason` du registre peut être remplie à la main ; elle est conservée.

Colonnes du tableau :
    min_manq       minutes avec trades chez Binance, absentes de nos données
    jours_manq     dont journées entières manquantes (fichier mensuel incomplet)
    min_trop       minutes présentes chez nous, absentes des bougies officielles
    identiques     part des minutes communes strictement identiques (OHLC et volume)
    vol_manq       volume manquant par rapport à l'officiel, sur le mois
    derive_max     plus grand écart de volume CUMULÉ, en part du volume du mois : un trade
                   rangé dans la minute voisine se compense aussitôt, un trade perdu persiste
    compensés      part des minutes en excès compensées exactement par une minute voisine
                   (proche de 100 % = simples déplacements de trades à la frontière des minutes)
    prix_hors      minutes où notre high / low sort de celui des minutes officielles voisines :
                   doit être 0, sinon erreur de reconstruction
    open/.../volume   part des minutes différentes pour chaque colonne

Le tableau complet est aussi écrit dans data/diagnostic_perp.csv.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from gaps import (
    MISMATCH_COLUMNS, data_dir, files_of, known_mismatches_path, load_known_mismatches, month_of,
)

COLS = ["open", "high", "low", "close", "volume"]


def to_minutes(s: pd.DataFrame) -> pd.DataFrame:
    return s.groupby(s["open_time"].dt.floor("min")).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"),
    )


def diagnose(path_1s, path_1m) -> dict:
    ours = to_minutes(pd.read_parquet(path_1s))
    official = pd.read_parquet(path_1m)
    official = official[official["volume"] > 0].set_index("open_time")[COLS]

    missing = official.index.difference(ours.index)
    extra = ours.index.difference(official.index)
    per_day = pd.Series(missing.floor("D")).value_counts()
    common = ours.index.intersection(official.index)
    a, b = ours.loc[common], official.loc[common]

    row = {
        "mois": f"{month_of(path_1s):%Y-%m}",
        "min_manq": len(missing),
        "jours_manq": int((per_day >= 1440).sum()),
        "min_trop": len(extra),
    }
    same = np.ones(len(common), bool)
    diffs = {}
    for col in COLS:
        eq = np.isclose(a[col], b[col], rtol=1e-9, atol=1e-9)
        diffs[col] = 1 - eq.mean()
        same &= eq
    row["identiques"] = same.mean()
    row["vol_manq"] = 1 - a["volume"].sum() / b["volume"].sum()

    ours = ours[ours.index.isin(official.index)]   # minutes absentes de l'officiel : rien à comparer
    idx = ours.index.union(official.index)
    diff = ours["volume"].reindex(idx, fill_value=0) - official["volume"].reindex(idx, fill_value=0)
    row["derive_max"] = diff.cumsum().abs().max() / official["volume"].sum()
    excess = diff[diff > 1e-6]
    comp = (np.isclose(excess, -diff.shift(-1).reindex(excess.index), atol=1e-6)
            | np.isclose(excess, -diff.shift(1).reindex(excess.index), atol=1e-6))
    row["compensés"] = comp.mean() if len(excess) else 1.0

    grid = official.reindex(idx)
    high_ref = grid["high"].rolling(3, center=True, min_periods=1).max().reindex(ours.index)
    low_ref = grid["low"].rolling(3, center=True, min_periods=1).min().reindex(ours.index)
    row["prix_hors"] = int(((ours["high"] > high_ref * (1 + 1e-12)) | (ours["low"] < low_ref * (1 - 1e-12))).sum())
    row.update(diffs)
    row["_mismatches"] = pd.concat([
        pd.DataFrame({"side": "absente_chez_nous", "minute": missing,
                      "official_volume": official.loc[missing, "volume"].values}),
        pd.DataFrame({"side": "absente_chez_binance", "minute": extra, "official_volume": 0.0}),
    ], ignore_index=True)
    return row


def register(mismatches: pd.DataFrame) -> None:
    """Écrit le registre des minutes irréparables, en conservant les `reason` déjà saisies."""
    known = load_known_mismatches()
    reasons = {(r.side, r.minute): r.reason for r in known.itertuples()}
    mismatches["reason"] = [reasons.get((r.side, r.minute), "") for r in mismatches.itertuples()]
    # les zones invalides sont saisies à la main : on les conserve telles quelles
    manual = known[known["side"] == "zone_invalide"]
    mismatches = pd.concat([mismatches, manual], ignore_index=True)
    mismatches["minute"] = pd.to_datetime(mismatches["minute"], utc=True)
    path = known_mismatches_path()
    mismatches[MISMATCH_COLUMNS].sort_values("minute").to_csv(path, index=False)
    print(f"{len(mismatches)} minute(s) enregistrée(s) dans {path}")
    if not mismatches.empty:
        print(mismatches.groupby([mismatches["minute"].dt.date, "side"])
              .agg(minutes=("minute", "size"), volume_officiel=("official_volume", "sum")).to_string())


def main() -> None:
    args = sys.argv[1:]
    do_register = "--register" in args
    wanted = {a for a in args if a != "--register"}
    f1s = {f"{month_of(p):%Y-%m}": p for p in files_of("futures_klines_1s")}
    f1m = {f"{month_of(p):%Y-%m}": p for p in files_of("futures_klines_1m")}
    months = sorted(f1s.keys() & f1m.keys())
    if wanted:
        months = [m for m in months if m in wanted]
    if not months:
        print("Aucun mois commun entre futures_klines_1s et futures_klines_1m.")
        return

    rows = []
    for m in months:
        rows.append(diagnose(f1s[m], f1m[m]))
        print(f"\r{m} analysé", end="", flush=True)
    print("\n")

    mismatches = pd.concat([r.pop("_mismatches") for r in rows], ignore_index=True)
    df = pd.DataFrame(rows)
    out = data_dir().parent / "diagnostic_perp.csv"
    df.to_csv(out, index=False)

    pct = ["identiques", "vol_manq", "compensés"] + COLS
    shown = df.copy()
    for c in pct:
        shown[c] = shown[c].map(lambda v: f"{v:.2%}")
    shown["derive_max"] = shown["derive_max"].map(lambda v: f"{v:.4%}")
    print(shown.to_string(index=False))
    print(f"\nTableau complet : {out}")
    print(f"Mois avec des prix hors des bornes officielles (erreur probable) : "
          f"{df.loc[df['prix_hors'] > 0, 'mois'].tolist() or 'aucun'}")
    if do_register:
        if wanted:
            print("\n--register s'utilise sans sélection de mois : le registre couvre tout l'historique.")
        else:
            print()
            register(mismatches)


if __name__ == "__main__":
    main()