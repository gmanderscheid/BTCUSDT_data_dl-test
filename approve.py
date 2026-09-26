"""
Approbation des entrées des registres d'exceptions, après examen.

Une anomalie détectée (statut « candidate ») n'est PAS une exception acceptée : les tests
échouent tant qu'elle n'a pas été examinée. Ce script la passe au statut « approved »
avec une raison obligatoire. La raison doit dire ce qui a été vérifié, par exemple
« maintenance annoncée par Binance le 25/04/2021 » ou « absente aussi du fichier
journalier, 5 min, 12 000 BTC, trou de la source ».

Usage
-----
    python approve.py --list                                   # entrées à examiner
    python approve.py gaps 2021-04-25 "maintenance Binance annoncée"
    python approve.py gaps 2021-04-25 "..." --dataset futures_klines_1s
    python approve.py minutes 2024-10-28 "bougies 1 min officielles absentes, fichier journalier inclus"
    python approve.py minutes 2024-10-28 "..." --side absente_chez_binance

Seules les entrées candidates de la date donnée sont modifiées. Pour les trous, la date
est celle du début du trou.
"""
from __future__ import annotations

import argparse

import pandas as pd

from gaps import known_gaps_path, known_mismatches_path, load_known_gaps, load_known_mismatches


def list_candidates() -> None:
    gaps = load_known_gaps()
    mism = load_known_mismatches()
    cg = gaps[gaps["status"] != "approved"]
    cm = mism[mism["status"] != "approved"]
    print(f"Trous longs à examiner : {len(cg)}")
    if not cg.empty:
        print(cg.assign(jour=cg["start"].dt.date).groupby(["jour", "dataset"])
              .agg(trous=("start", "size"), duree_s=("duration_s", "sum")).to_string())
    print(f"\nMinutes du perpétuel à examiner : {len(cm)}")
    if not cm.empty:
        print(cm.assign(jour=cm["minute"].dt.date).groupby(["jour", "side"])
              .agg(minutes=("minute", "size"), volume_officiel=("official_volume", "sum")).to_string())


def approve(registry: str, day: str, reason: str, dataset: str | None, side: str | None) -> None:
    if not reason.strip():
        raise SystemExit("La raison est obligatoire.")
    d = pd.Timestamp(day).date()
    if registry == "gaps":
        path, df = known_gaps_path(), load_known_gaps()
        mask = df["start"].dt.date == d
        if dataset:
            mask &= df["dataset"] == dataset
    else:
        path, df = known_mismatches_path(), load_known_mismatches()
        mask = df["minute"].dt.date == d
        if side:
            mask &= df["side"] == side
    mask &= df["status"] != "approved"
    if not mask.any():
        raise SystemExit(f"Aucune entrée candidate le {d} dans {path.name}.")
    df.loc[mask, "status"] = "approved"
    df.loc[mask, "reason"] = reason
    df.to_csv(path, index=False)
    print(f"{int(mask.sum())} entrée(s) approuvée(s) le {d} dans {path.name}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("registry", nargs="?", choices=["gaps", "minutes"])
    parser.add_argument("day", nargs="?", help="date AAAA-MM-JJ")
    parser.add_argument("reason", nargs="?", help="ce qui a été vérifié")
    parser.add_argument("--dataset", help="limiter aux trous d'un dataset (registre gaps)")
    parser.add_argument("--side", help="limiter à un type de minute (registre minutes)")
    parser.add_argument("--list", action="store_true", help="liste les entrées à examiner")
    args = parser.parse_args()

    if args.list or not args.registry:
        list_candidates()
        return
    if not (args.day and args.reason):
        parser.error("indique la date et la raison")
    approve(args.registry, args.day, args.reason, args.dataset, args.side)


if __name__ == "__main__":
    main()
