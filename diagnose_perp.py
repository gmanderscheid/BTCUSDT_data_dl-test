"""
Diagnostic du perpétuel : compare, mois par mois, nos bougies 1 s reconstruites (agrégées
en 1 min) aux bougies 1 min officielles de Binance.

Deux vues, calculées avec les MÊMES fonctions et les MÊMES seuils que pytest
(quality_checks.py) :
- vue BRUTE (min_manq, jours_manq, min_trop) : sans aucune exception, pour voir ce que les
  données contiennent vraiment ;
- vue FILTRÉE (toutes les autres colonnes) : après les exceptions approuvées du registre,
  c'est-à-dire ce que pytest juge. La colonne « erreurs » compte les contrôles pytest en
  échec pour le mois, et leur détail est affiché sous le tableau.

Usage :
    python diagnose_perp.py                 # tous les mois
    python diagnose_perp.py 2021-03 2024-10 # seulement ces mois
    python diagnose_perp.py --register      # inscrit les minutes manquantes restantes comme
                                            # CANDIDATES dans data/known_minute_mismatches.csv

À n'utiliser qu'APRÈS download_binance.py, qui tente de réparer ces minutes avec les
fichiers journaliers. --register ne fait que DÉTECTER : chaque entrée doit ensuite être
examinée (le volume officiel concerné est affiché) puis approuvée avec approve.py. Les
statuts, les raisons et les zones invalides saisies à la main sont conservés.

Colonnes :
    min_manq / jours_manq   minutes officielles absentes chez nous (dont journées entières), brut
    min_trop                nos minutes absentes des bougies officielles, brut
    comparées               part des minutes officielles comparées après exceptions
    vol_exclu               part du volume officiel exclu par les exceptions
    ecart_vol               écart du volume total, minutes communes
    derive_max              plus grand écart de volume cumulé, part du volume du mois
    compensés               part des minutes en excès compensées exactement par une voisine
    close_diff              part des minutes au close différent
    erreurs                 nombre de contrôles pytest en échec
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import data_contract as contract
import quality_checks as qc
from gaps import (
    MISMATCH_COLUMNS, data_dir, files_of, known_mismatches_path, load_known_mismatches, month_of,
    next_month, utc,
)

T = qc.THRESHOLDS


def compensation_share(ours: pd.DataFrame, official: pd.DataFrame) -> float:
    idx = ours.index.union(official.index)
    diff = ours["volume"].reindex(idx, fill_value=0) - official["volume"].reindex(idx, fill_value=0)
    excess = diff[diff > 1e-6]
    if not len(excess):
        return 1.0
    comp = (np.isclose(excess, -diff.shift(-1).reindex(excess.index), atol=1e-6)
            | np.isclose(excess, -diff.shift(1).reindex(excess.index), atol=1e-6))
    return float(comp.mean())


def diagnose(path_1s, path_1m, registry: pd.DataFrame) -> dict:
    m = month_of(path_1s)
    start, end = utc(m), utc(next_month(m))
    ours_full = qc.to_minutes(pd.read_parquet(path_1s))
    official_full = qc.official_minutes(pd.read_parquet(path_1m))

    missing, extra = qc.missing_minutes(ours_full, official_full)
    per_day = pd.Series(missing.floor("D")).value_counts()
    ours, official, errors, stats = qc.apply_registry(ours_full, official_full, registry, start, end)

    # mêmes contrôles et mêmes seuils que TestPerpetualVsOfficial1m
    if stats["compared_share"] < contract.MIN_COMPARED_SHARE:
        errors.append(f"minutes comparées {stats['compared_share']:.2%}")
    if stats["excluded_official_volume_share"] > contract.MAX_EXCLUDED_VOLUME_SHARE:
        errors.append(f"volume exclu {stats['excluded_official_volume_share']:.3%}")
    m_left, e_left = qc.missing_minutes(ours, official)
    if len(m_left):
        errors.append(f"{len(m_left)} minute(s) officielle(s) absente(s) chez nous")
    if len(e_left):
        errors.append(f"{len(e_left)} minute(s) absente(s) de l'officiel")
    errors += qc.price_errors(ours, official, T["MAX_PRICE_EXCESS"], ours_full)
    close = qc.close_mismatch_share(ours, official)
    if close > T["MAX_CLOSE_MISMATCH"]:
        errors.append(f"close différent sur {close:.3%} des minutes")
    errors += qc.drift_errors(ours, official, T["MAX_VOLUME_DRIFT"])
    errors += qc.local_volume_errors(ours, official, T["MAX_DAY_VOLUME_GAP"],
                                     T["MAX_HOUR_VOLUME_GAP"], T["HOUR_VOLUME_FLOOR"])
    gap = qc.month_volume_gap(ours, official)
    if gap > T["MAX_VOLUME_DEFICIT"]:
        errors.append(f"écart de volume du mois {gap:.3%}")

    idx = ours.index.union(official.index)
    diff = ours["volume"].reindex(idx, fill_value=0) - official["volume"].reindex(idx, fill_value=0)
    total = official["volume"].sum()
    return {
        "mois": f"{m:%Y-%m}",
        "min_manq": len(missing),
        "jours_manq": int((per_day >= 1440).sum()),
        "min_trop": len(extra),
        "comparées": stats["compared_share"],
        "vol_exclu": stats["excluded_official_volume_share"],
        "ecart_vol": gap,
        "derive_max": float(diff.cumsum().abs().max() / total) if total else 0.0,
        "compensés": compensation_share(ours, official),
        "close_diff": close,
        "erreurs": len(errors),
        "_detail": errors,
        "_mismatches": pd.concat([
            pd.DataFrame({"side": "absente_chez_nous", "minute": missing,
                          "official_volume": official_full.loc[missing, "volume"].values}),
            pd.DataFrame({"side": "absente_chez_binance", "minute": extra, "official_volume": 0.0}),
        ], ignore_index=True),
    }


def register(mismatches: pd.DataFrame) -> None:
    """
    Écrit le registre des minutes. Les nouvelles entrées sont CANDIDATES ; les entrées déjà
    connues gardent leur statut et leur raison ; les zones invalides saisies à la main sont
    conservées. Les entrées qui ne sont plus observées sont retirées et signalées.
    """
    known = load_known_mismatches()
    review = {(r.side, r.minute): (r.status, r.reason) for r in known.itertuples()}
    mismatches["minute"] = pd.to_datetime(mismatches["minute"], utc=True)
    mismatches["status"] = [review.get((r.side, r.minute), ("candidate", ""))[0] for r in mismatches.itertuples()]
    mismatches["reason"] = [review.get((r.side, r.minute), ("candidate", ""))[1] for r in mismatches.itertuples()]
    manual = known[known["side"] == "zone_invalide"]
    out = pd.concat([mismatches, manual], ignore_index=True)
    out["minute"] = pd.to_datetime(out["minute"], utc=True)
    out = out[MISMATCH_COLUMNS].sort_values("minute")

    keys = set(zip(out["side"], out["minute"]))
    removed = known[[(r.side, r.minute) not in keys for r in known.itertuples()]]
    path = known_mismatches_path()
    out.to_csv(path, index=False)
    cand = out[out["status"] != "approved"]
    print(f"{len(out)} minute(s) dans {path}, dont {len(cand)} à examiner")
    if not cand.empty:
        print(cand.groupby([cand["minute"].dt.date, "side"])
              .agg(minutes=("minute", "size"), volume_officiel=("official_volume", "sum")).to_string())
        print("\nExamine chaque journée (volume officiel faible : liquidations plausibles ; volume de "
              "marché normal : trou de la source), puis : python approve.py minutes <date> \"<raison>\"")
    if not removed.empty:
        print(f"\n{len(removed)} entrée(s) retirée(s) car plus observée(s).")


def main() -> None:
    args = sys.argv[1:]
    do_register = "--register" in args
    wanted = {a for a in args if a != "--register"}
    f1s = {f"{month_of(p):%Y-%m}": p for p in files_of("futures_klines_1s")}
    f1m = {f"{month_of(p):%Y-%m}": p for p in files_of("futures_klines_1m")}
    missing_ref = sorted(set(f1s) - set(f1m))
    months = sorted(f1s.keys() & f1m.keys())
    if wanted:
        months = [m for m in months if m in wanted]
    if missing_ref:
        print(f"ATTENTION : mois sans référence 1 min, non diagnostiqués : {missing_ref}\n")
    if not months:
        print("Aucun mois commun entre futures_klines_1s et futures_klines_1m.")
        return

    registry = load_known_mismatches()
    rows = []
    for m in months:
        rows.append(diagnose(f1s[m], f1m[m], registry))
        print(f"\r{m} analysé", end="", flush=True)
    print("\n")

    mismatches = pd.concat([r.pop("_mismatches") for r in rows], ignore_index=True)
    details = {r["mois"]: r.pop("_detail") for r in rows}
    df = pd.DataFrame(rows)
    out = data_dir().parent / "diagnostic_perp.csv"
    df.to_csv(out, index=False)

    shown = df.copy()
    for c in ("comparées", "vol_exclu", "ecart_vol", "compensés", "close_diff"):
        shown[c] = shown[c].map(lambda v: f"{v:.2%}")
    shown["derive_max"] = shown["derive_max"].map(lambda v: f"{v:.4%}")
    print(shown.to_string(index=False))
    print(f"\nTableau complet : {out}")
    failing = {m: d for m, d in details.items() if d}
    print(f"Mois en échec pour pytest : {list(failing) or 'aucun'}")
    for m, d in failing.items():
        print(f"  {m} : " + " | ".join(e.splitlines()[0] for e in d))

    if do_register:
        if wanted:
            print("\n--register s'utilise sans sélection de mois : le registre couvre tout l'historique.")
        else:
            print()
            register(mismatches)


if __name__ == "__main__":
    main()