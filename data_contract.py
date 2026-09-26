"""
Contrat de couverture des données : ce qui DOIT être présent pour qu'une validation soit
complète.

Pourquoi : sans contrat, un téléchargement partiel, un mauvais dossier de données ou la
disparition des fichiers de référence réduit silencieusement ce qui est contrôlé, et
pytest peut rester vert. Les tests de couverture (catégorie `couverture`) comparent les
fichiers présents à ce contrat.

Modes (variable d'environnement VALIDATION_MODE) :
- "complete" (défaut) : tout écart au contrat fait échouer la validation ;
- "partial" : exploration volontaire sur un sous-ensemble ; les tests de couverture sont
  ignorés et un avertissement le signale. Un résultat « partial » ne vaut PAS validation.
"""
from __future__ import annotations

import os
from datetime import date, timedelta

# premier mois attendu pour chaque dataset requis
REQUIRED_DATASETS: dict[str, str] = {
    "futures_klines_1s": "2020-01",
    "futures_klines_1m": "2020-01",
    "futures_funding": "2020-01",
    "spot_klines_1s": "2020-01",
}

# datasets publiés chaque jour (fichiers journaliers) : le mois en cours doit exister
DAILY_DATASETS = {"futures_klines_1s", "futures_klines_1m", "spot_klines_1s"}

# la vérification du perpétuel exige, pour chaque mois de 1 s, sa référence 1 min
REFERENCE_OF = {"futures_klines_1s": "futures_klines_1m"}

# part minimale des minutes officielles effectivement comparées, après exclusions
MIN_COMPARED_SHARE = float(os.environ.get("MIN_COMPARED_SHARE", "0.99"))
# part maximale du volume officiel exclu des comparaisons par le registre, par mois
MAX_EXCLUDED_VOLUME_SHARE = float(os.environ.get("MAX_EXCLUDED_VOLUME_SHARE", "0.005"))


def validation_mode() -> str:
    mode = os.environ.get("VALIDATION_MODE", "complete")
    if mode not in ("complete", "partial"):
        raise ValueError(f"VALIDATION_MODE invalide : {mode!r} (complete ou partial)")
    return mode


def expected_last_month(dataset: str, today: date | None = None) -> date:
    """
    Dernier mois attendu : le mois en cours pour les datasets journaliers (à partir du 2
    du mois, la veille est publiée), sinon le mois précédent (funding, publié en fin de
    mois ; tolérance de 3 jours après le début du mois).
    """
    today = today or date.today()
    if dataset in DAILY_DATASETS:
        ref = today - timedelta(days=2)
        return ref.replace(day=1)
    ref = today - timedelta(days=3)
    return (ref.replace(day=1) - timedelta(days=1)).replace(day=1)
