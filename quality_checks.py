"""
Contrôles de qualité des données, partagés par les tests (pytest) et le diagnostic.

Chaque fonction renvoie une liste d'erreurs lisibles : une liste vide signifie que le
contrôle est réussi. Les tests de données (tests/test_data_quality.py) les appliquent
aux vrais fichiers ; les tests unitaires (tests/test_pipeline.py) les appliquent à des
cas synthétiques, positifs et négatifs, pour vérifier que les contrôles détectent bien
ce qu'ils prétendent détecter. diagnose_perp.py réutilise les mêmes calculs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import os

from gaps import MISMATCH_SIDES, STATUSES, approved

ONE_MIN = pd.Timedelta(minutes=1)
ONE_S = pd.Timedelta(seconds=1)

# Seuils partagés par pytest et diagnose_perp.py (surchargeables par variable d'environnement).
_DEFAULTS = {
    "MAX_SHORT_GAP_RATIO": 0.005, "MAX_SHORT_GAP_RATIO_PERP": 0.60,
    "MAX_PRICE_JUMP": 0.10, "MAX_PRICE_JUMP_PERP": 0.15,
    "MAX_VWAP_TOL": 0.001, "MAX_PRICE_EXCESS": 0.0005, "MAX_CLOSE_MISMATCH": 0.005,
    "MAX_VOLUME_DRIFT": 0.001, "MAX_DAY_VOLUME_GAP": 0.005, "MAX_HOUR_VOLUME_GAP": 0.05,
    "HOUR_VOLUME_FLOOR": 100.0, "MAX_VOLUME_DEFICIT": 0.01,
    "MAX_FUNDING_ABS": 0.03, "MAX_FUNDING_MEDIAN": 0.001,
}
THRESHOLDS = {k: float(os.environ.get(k, v)) for k, v in _DEFAULTS.items()}


def _examples(index, n: int = 3) -> list:
    return list(index[:n])


# =========================================================================== bougies 1 s

def finite_errors(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Colonnes numériques finies (ni NaN, ni ±inf)."""
    errors = []
    for col in cols:
        bad = ~np.isfinite(df[col].to_numpy(dtype="float64"))
        if bad.any():
            errors.append(f"{col} : {bad.sum()} valeur(s) non finie(s)")
    return errors


def integer_errors(df: pd.DataFrame, col: str) -> list[str]:
    """Colonne de comptage : type entier (donc finie et sans partie décimale)."""
    if not pd.api.types.is_integer_dtype(df[col]):
        return [f"{col} n'est pas de type entier ({df[col].dtype})"]
    return []


def duration_errors(df: pd.DataFrame, next_file_first: pd.Timestamp | None = None) -> list[str]:
    """
    Durée des bougies 1 s : open_time <= close_time < open_time + 1 s.

    Fin normale : open_time + 999 ms (ms) ou + 999,999 ms (µs, spot depuis 2025).
    Bougie TRONQUÉE (close_time avant la fin normale) : acceptée seulement si la seconde
    suivante est absente, y compris quand elle se trouverait dans le fichier du mois
    suivant (next_file_first = premier open_time de ce fichier).
    """
    errors = []
    delta = df["close_time"] - df["open_time"]
    negative = delta < pd.Timedelta(0)
    too_long = delta >= ONE_S
    truncated = (delta >= pd.Timedelta(0)) & (delta < pd.Timedelta(milliseconds=999))

    present = pd.DatetimeIndex(df["open_time"])
    if next_file_first is not None:
        present = present.append(pd.DatetimeIndex([next_file_first]))
    next_missing = ~(df["open_time"] + ONE_S).isin(present)
    unexplained = truncated & ~next_missing

    def show(mask):
        return df.loc[mask, ["open_time", "close_time"]].head(3).to_string(index=False)

    if negative.any():
        errors.append(f"{negative.sum()} bougie(s) avec close_time < open_time :\n{show(negative)}")
    if too_long.any():
        errors.append(f"{too_long.sum()} bougie(s) de plus d'une seconde :\n{show(too_long)}")
    if unexplained.any():
        errors.append(f"{unexplained.sum()} bougie(s) tronquée(s) sans arrêt du marché juste après :\n{show(unexplained)}")
    return errors


def price_jump_errors(df: pd.DataFrame, limit: float) -> list[str]:
    """
    Variation SIMPLE du prix de clôture entre deux secondes consécutives :
    |close_t / close_{t-1} - 1| <= limit, symétrique (+10 % et -10 % pour limit = 0,10).
    Seules les bougies séparées d'exactement une seconde sont comparées. Ce contrôle porte
    sur les clôtures ; les mèches à l'intérieur d'une seconde sont contrôlées par OHLC.
    """
    consecutive = df["open_time"].diff() == ONE_S
    ret = (df["close"] / df["close"].shift() - 1).abs()
    jumps = df.loc[consecutive & (ret > limit), "open_time"]
    if jumps.empty:
        return []
    return [f"{len(jumps)} saut(s) de prix > {limit:.0%} en 1 s, ex. {jumps.head(3).tolist()}"]


def vwap_errors(df: pd.DataFrame, tol: float) -> list[str]:
    """
    Prix moyens implicites dans la fourchette de la bougie, à `tol` près :
    - quote_volume / volume (tous les trades) ;
    - taker_buy_quote / taker_buy_base (trades à l'initiative de l'acheteur).
    La marge `tol` est une marge de prudence pour les arrondis des volumes publiés ; elle
    n'a pas été calibrée sur des cas réels.
    """
    errors = []
    for label, q, b in (("VWAP", "quote_volume", "volume"),
                        ("VWAP acheteur", "taker_buy_quote", "taker_buy_base")):
        v = df[df[b] > 0]
        vwap = v[q] / v[b]
        bad = (vwap < v["low"] * (1 - tol)) | (vwap > v["high"] * (1 + tol))
        if bad.any():
            errors.append(f"{label} hors de [low, high] (±{tol:.2%}) : {bad.sum()} bougie(s), "
                          f"ex. {v.loc[bad, 'open_time'].head(3).tolist()}")
    return errors


def zero_consistency_errors(df: pd.DataFrame) -> list[str]:
    """Les volumes nuls vont ensemble : base nulle <=> quote nul, pour le total et l'acheteur."""
    errors = []
    for b, q in (("volume", "quote_volume"), ("taker_buy_base", "taker_buy_quote")):
        bad = (df[b] == 0) != (df[q] == 0)
        if bad.any():
            errors.append(f"{b} et {q} incohérents (l'un nul, pas l'autre) : {bad.sum()} bougie(s)")
    bad = (df["n_trades"] > 0) & (df["volume"] <= 0)
    if bad.any():
        errors.append(f"{bad.sum()} bougie(s) avec des trades mais un volume nul")
    return errors


# =========================================================================== perpétuel vs 1 min officiel

MINUTE_COLS = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base"]


def to_minutes(s: pd.DataFrame) -> pd.DataFrame:
    """Agrège des bougies 1 s en bougies 1 min (index = début de minute, UTC)."""
    return s.groupby(s["open_time"].dt.floor("min")).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), quote_volume=("quote_volume", "sum"),
        taker_buy_base=("taker_buy_base", "sum"),
    )


def official_minutes(m: pd.DataFrame) -> pd.DataFrame:
    """Bougies 1 min officielles avec des trades, indexées par début de minute."""
    return m[m["volume"] > 0].set_index("open_time")[MINUTE_COLS]


def registry_errors(registry: pd.DataFrame) -> list[str]:
    """Structure du registre des minutes : types, alignement, doublons, statut, raison."""
    errors = []
    if registry.empty:
        return errors
    bad_side = ~registry["side"].isin(MISMATCH_SIDES)
    if bad_side.any():
        errors.append(f"side invalide : {sorted(registry.loc[bad_side, 'side'].unique())}")
    misaligned = registry["minute"] != registry["minute"].dt.floor("min")
    if misaligned.any():
        errors.append(f"{misaligned.sum()} minute(s) non alignée(s) sur la minute")
    dup = registry.duplicated(["side", "minute"])
    if dup.any():
        errors.append(f"{dup.sum()} entrée(s) en double (side, minute)")
    conflict = registry.groupby("minute")["side"].nunique() > 1
    if conflict.any():
        errors.append(f"{conflict.sum()} minute(s) inscrite(s) avec plusieurs side, ex. {_examples(conflict[conflict].index)}")
    bad_status = ~registry["status"].isin(STATUSES)
    if bad_status.any():
        errors.append(f"statut invalide : {sorted(registry.loc[bad_status, 'status'].unique())}")
    no_reason = (registry["status"] == "approved") & (registry["reason"].str.strip() == "")
    if no_reason.any():
        errors.append(f"{no_reason.sum()} entrée(s) approuvée(s) sans raison")
    return errors


def apply_registry(ours: pd.DataFrame, official: pd.DataFrame, registry: pd.DataFrame,
                   start: pd.Timestamp, end: pd.Timestamp):
    """
    Applique les exceptions APPROUVÉES du registre des minutes à un mois [start, end[.

    Renvoie (ours, official, errors, stats) :
    - ours / official privés des minutes exclues ;
    - errors : entrées candidates (non examinées) et entrées approuvées obsolètes, qui ne
      correspondent plus aux données ;
    - stats : part de minutes officielles comparées, volume officiel exclu.
    """
    reg = registry[(registry["minute"] >= start) & (registry["minute"] < end)]
    errors = []
    cand = reg[reg["status"] != "approved"]
    if not cand.empty:
        errors.append(f"{len(cand)} minute(s) du registre non examinée(s) (candidate), ex. "
                      f"{_examples(pd.DatetimeIndex(cand['minute']))} : python approve.py minutes <date> \"<raison>\"")
    ok = approved(reg)
    nous = pd.DatetimeIndex(ok.loc[ok["side"] == "absente_chez_nous", "minute"])
    binance = pd.DatetimeIndex(ok.loc[ok["side"] == "absente_chez_binance", "minute"])
    invalid = pd.DatetimeIndex(ok.loc[ok["side"] == "zone_invalide", "minute"])

    stale_nous = nous[nous.isin(ours.index)]
    stale_binance = binance[binance.isin(official.index)]
    if len(stale_nous) or len(stale_binance):
        errors.append(f"exception(s) obsolète(s) : {len(stale_nous)} minute(s) 'absente_chez_nous' désormais "
                      f"présente(s) chez nous, {len(stale_binance)} minute(s) 'absente_chez_binance' désormais "
                      f"présente(s) chez Binance. Relance python diagnose_perp.py --register.")

    total_off = official["volume"].sum()
    excl_off = official.index.isin(nous.union(invalid))
    stats = {
        "official_minutes": len(official),
        "excluded_official_volume_share": float(official.loc[excl_off, "volume"].sum() / total_off) if total_off else 0.0,
    }
    official = official[~excl_off]
    ours = ours[~ours.index.isin(binance.union(invalid))]
    stats["compared_minutes"] = int(official.index.isin(ours.index).sum())
    stats["compared_share"] = stats["compared_minutes"] / stats["official_minutes"] if stats["official_minutes"] else 0.0
    return ours, official, errors, stats


def missing_minutes(ours: pd.DataFrame, official: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """(minutes officielles absentes chez nous, nos minutes absentes de l'officiel)."""
    return official.index.difference(ours.index), ours.index.difference(official.index)


def _neighbour_extreme(frame: pd.DataFrame, at: pd.DatetimeIndex, col: str, how: str) -> pd.Series:
    """Max (ou min) de `col` aux timestamps EXACTS t - 1 min, t, t + 1 min (absents ignorés)."""
    vals = [frame[col].reindex(at + k * ONE_MIN).to_numpy() for k in (-1, 0, 1)]
    stacked = np.vstack(vals)
    with np.errstate(all="ignore"):
        out = np.nanmax(stacked, axis=0) if how == "max" else np.nanmin(stacked, axis=0)
    return pd.Series(out, index=at)


def price_errors(ours: pd.DataFrame, official: pd.DataFrame, tol: float,
                 ours_full: pd.DataFrame | None = None) -> list[str]:
    """
    Prix extrêmes cohérents avec les bougies officielles, à `tol` près, dans les deux sens.

    Les voisines sont les minutes EXACTES t - 1, t, t + 1 (pas les lignes voisines, qui
    peuvent être à plusieurs heures après un trou). Un trade peut être rangé dans la minute
    d'à côté, d'où la fenêtre de 3 minutes :
    - trop large : notre high(t) > max des high officiels en t-1, t, t+1 (idem low) ;
    - mèche disparue : le high officiel en t n'apparaît pas dans nos high en t-1, t, t+1
      (idem low) ;
    - un contrôle n'est fait que si sa fenêtre est complète : si une voisine a des trades
      d'un côté mais pas de l'autre (trou d'un des deux fichiers), la minute n'est pas
      vérifiable et elle est ignorée plutôt que jugée sur une référence tronquée.
    """
    errors = []
    ours_full = ours if ours_full is None else ours_full
    common = ours.index.intersection(official.index)

    def verifiable(at: pd.DatetimeIndex) -> np.ndarray:
        ok = np.ones(len(at), dtype=bool)
        for k in (-1, 1):
            nb = at + k * ONE_MIN
            ok &= nb.isin(ours_full.index) == nb.isin(official.index)
        return ok

    at = common[verifiable(common)]
    hi_ref = _neighbour_extreme(official, at, "high", "max")
    lo_ref = _neighbour_extreme(official, at, "low", "min")
    too_high = ours.loc[at, "high"].to_numpy() > hi_ref.to_numpy() * (1 + tol)
    too_low = ours.loc[at, "low"].to_numpy() < lo_ref.to_numpy() * (1 - tol)
    hi_ours = _neighbour_extreme(ours, at, "high", "max")
    lo_ours = _neighbour_extreme(ours, at, "low", "min")
    lost_high = official.loc[at, "high"].to_numpy() > hi_ours.to_numpy() * (1 + tol)
    lost_low = official.loc[at, "low"].to_numpy() < lo_ours.to_numpy() * (1 - tol)

    for label, bad in (("high au-dessus des voisines officielles", too_high),
                       ("low en dessous des voisines officielles", too_low),
                       ("high officiel absent de nos données (mèche disparue)", lost_high),
                       ("low officiel absent de nos données (mèche disparue)", lost_low)):
        if bad.any():
            errors.append(f"{label} : {bad.sum()} minute(s), ex. {_examples(at[bad])}")
    skipped = len(common) - len(at)
    if skipped > 0.01 * max(len(common), 1):
        errors.append(f"{skipped} minute(s) non vérifiables (voisine absente d'un seul côté), plus de 1 %")
    return errors


def close_mismatch_share(ours: pd.DataFrame, official: pd.DataFrame) -> float:
    """Part des minutes communes dont le close diffère de l'officiel."""
    common = ours.index.intersection(official.index)
    if not len(common):
        return 0.0
    eq = np.isclose(ours.loc[common, "close"], official.loc[common, "close"], rtol=1e-12, atol=0)
    return float(1 - eq.mean())


def drift_errors(ours: pd.DataFrame, official: pd.DataFrame, limit: float) -> list[str]:
    """
    Écart CUMULÉ de volume (et de volume acheteur) : max |Σ (nous - officiel)| <= limit x
    volume officiel du mois. Un trade rangé dans la minute voisine se compense aussitôt ;
    un trade perdu ou en double crée un écart qui persiste. Une valeur non finie échoue.
    """
    idx = ours.index.union(official.index)
    total = official["volume"].sum()
    errors = []
    if not np.isfinite(total) or total <= 0:
        return [f"volume officiel du mois invalide ({total})"]
    for col in ("volume", "taker_buy_base"):
        if not (np.isfinite(ours[col]).all() and np.isfinite(official[col]).all()):
            errors.append(f"{col} : dérive non calculable (valeurs non finies)")
            continue
        diff = ours[col].reindex(idx, fill_value=0) - official[col].reindex(idx, fill_value=0)
        drift = diff.cumsum().abs()
        worst = drift.max()
        if not np.isfinite(worst):
            errors.append(f"{col} : dérive non calculable (valeurs non finies)")
        elif worst > limit * total:
            errors.append(f"{col} : dérive max {worst:.3f} BTC ({worst / total:.4%} du volume du mois, "
                          f"max {limit:.4%}) le {drift.idxmax()}")
    return errors


def local_volume_errors(ours: pd.DataFrame, official: pd.DataFrame, day_tol: float,
                        hour_tol: float, hour_floor: float) -> list[str]:
    """
    Écarts de volume LOCAUX, que la dérive mensuelle laisse passer quand ils sont petits
    rapportés au mois (une heure amputée de 30 % ne pèse que 0,04 % d'un mois) :
    - par jour : |nous - officiel| <= day_tol x officiel ;
    - par heure : |nous - officiel| <= max(hour_tol x officiel, hour_floor BTC).
    Le plancher en BTC absorbe les trades déplacés à la frontière d'une heure, observés
    jusqu'à 60 BTC. Même contrôle pour le volume acheteur.
    """
    errors = []
    for col in ("volume", "taker_buy_base"):
        for label, freq, tol, floor in (("jour", "D", day_tol, 0.0), ("heure", "h", hour_tol, hour_floor)):
            a = ours[col].groupby(ours.index.floor(freq)).sum()
            b = official[col].groupby(official.index.floor(freq)).sum()
            idx = a.index.union(b.index)
            a, b = a.reindex(idx, fill_value=0), b.reindex(idx, fill_value=0)
            diff = (a - b).abs()
            bad = diff > np.maximum(tol * b, floor)
            if bad.any():
                worst = (diff / b.replace(0, np.nan))[bad].sort_values(ascending=False)
                errors.append(f"{col} par {label} : {bad.sum()} période(s) hors tolérance, pire "
                              f"{worst.iloc[0]:.2%} le {worst.index[0]}" if len(worst.dropna())
                              else f"{col} par {label} : {bad.sum()} période(s) hors tolérance")
    return errors


def month_volume_gap(ours: pd.DataFrame, official: pd.DataFrame) -> float:
    """Écart relatif du volume total, calculé sur les MINUTES COMMUNES aux deux sources."""
    common = ours.index.intersection(official.index)
    b = official.loc[common, "volume"].sum()
    return float(abs(1 - ours.loc[common, "volume"].sum() / b)) if b else 0.0


# =========================================================================== funding

def funding_errors(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """
    Échéancier du funding d'un mois [start, end[ :
    - un seul intervalle déclaré sur le mois (sinon, à traiter explicitement) ;
    - les instants de paiement (arrondis à la minute) sont EXACTEMENT les multiples de
      l'intervalle depuis le début du mois : ni paiement manquant (au début, à la fin ou
      au milieu), ni paiement en trop, ni intervalle trop court.
    """
    if df.empty:
        return ["aucun paiement de funding"]
    intervals = df["funding_interval_hours"].dropna().unique()
    if len(intervals) != 1:
        return [f"intervalles déclarés multiples ou absents : {sorted(intervals)}"]
    step = pd.Timedelta(hours=float(intervals[0]))
    expected = pd.date_range(start, end, freq=step, inclusive="left")
    got = pd.DatetimeIndex(df["calc_time"].dt.round("min"))
    missing = expected.difference(got)
    extra = got.difference(expected)
    errors = []
    if len(missing):
        errors.append(f"{len(missing)} paiement(s) manquant(s) sur {len(expected)}, ex. {_examples(missing)}")
    if len(extra):
        errors.append(f"{len(extra)} paiement(s) hors échéancier, ex. {_examples(extra)}")
    dup = got.duplicated()
    if dup.any():
        errors.append(f"{dup.sum()} paiement(s) en double")
    return errors


def funding_rate_errors(df: pd.DataFrame, max_abs: float, max_median: float) -> list[str]:
    """
    Ordre de grandeur du taux : |taux| < max_abs sur chaque paiement, et médiane de |taux|
    <= max_median. La médiane attrape une erreur d'unité d'un facteur 100 (0,01 % lu comme
    1 %), qu'une borne maximale seule laisserait passer.
    """
    r = df["last_funding_rate"].abs()
    errors = []
    if (r >= max_abs).any():
        errors.append(f"funding rate aberrant : max {r.max():.4%} (limite {max_abs:.2%})")
    if r.median() > max_median:
        errors.append(f"médiane de |funding rate| {r.median():.4%} > {max_median:.3%} : erreur d'unité probable")
    return errors
