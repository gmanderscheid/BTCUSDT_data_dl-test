"""
Tests de qualité des DONNÉES téléchargées par download_binance.py.

Pourquoi tester des données ?
-----------------------------
Un modèle de trading n'est jamais meilleur que les données sur lesquelles il apprend.
Une erreur dans les données ne fait pas planter le code : elle produit silencieusement
un modèle qui semble excellent en backtest et qui perd de l'argent en réel. Par exemple :

- une seconde en double fait apparaître un rendement nul qui n'a pas existé ;
- un prix aberrant crée un « signal » que le modèle va apprendre par cœur ;
- un trou de 3 h comblé naïvement fait croire à un marché immobile ;
- une bougie rangée dans le mauvais mois fausse le découpage train / test.

Ces tests sont la première barrière entre les fichiers bruts et les modèles. Ils ne
remplacent pas une revue : ils vérifient des contrats précis, énoncés dans chaque
docstring. Un pytest vert signifie « tous les contrats énoncés sont respectés sur les
données présentes », et le contrat de couverture (data_contract.py) garantit que les
données présentes sont bien celles attendues.

Les contrôles eux-mêmes sont dans quality_checks.py. Leur capacité à détecter les
défauts est vérifiée par les tests unitaires de tests/test_pipeline.py (cas synthétiques
positifs et négatifs).

Lancement (depuis la racine du projet)
--------------------------------------
    pytest                          # tests unitaires + tests de données
    pytest -m unit                  # tests unitaires seulement (rapides, sans données)
    pytest -m "not unit"            # tests de données seulement
    pytest -k "2020-06"             # seulement un mois
    pytest --lf                     # seulement ce qui a échoué la dernière fois

Avant le premier lancement : `python gaps.py`, `python diagnose_perp.py --register`, puis
examiner et approuver les exceptions (`python approve.py --list`).

Catégories (option -m)
----------------------
    structure       fichier non vide, colonnes, types, valeurs finies, n_trades entier, pas de .tmp
    chronologie     doublons, tri, bon mois, alignement, durée des bougies (y compris entre deux mois)
    completude      contrat de couverture (datasets, premier / dernier mois, mois manquants,
                    fraîcheur, référence 1 min), trous courts / longs, échéancier du funding
    coherence       prix positifs, OHLC, sauts de prix, ordre de grandeur du funding
    volumes         volumes positifs, acheteur <= total, nuls cohérents, VWAP et VWAP acheteur
    registre        structure des deux registres, exceptions examinées, pas d'exception obsolète
    verification    perpétuel 1 s comparé aux bougies 1 min officielles
    unit            tests unitaires du pipeline et des contrôles (tests/test_pipeline.py)

    Dataset         klines (1 s, spot et perpétuel), spot, perp (1 s + référence 1 min),
                    reference (bougies 1 min officielles), funding

Exemples :
    pytest -m "completude or chronologie"       # plusieurs catégories
    pytest -m "perp and verification"           # une catégorie sur un marché
    pytest -m registre                          # après une approbation dans un registre

Seuils (variables d'environnement)
----------------------------------
    VALIDATION_MODE             complete (défaut) ou partial (exploration, NE VAUT PAS validation)
    SHORT_GAP_MAX_SECONDS       durée max d'un trou « court » (60)
    MAX_SHORT_GAP_RATIO         part max de secondes en trous courts, spot (0.005)
    MAX_SHORT_GAP_RATIO_PERP    idem perpétuel, simple garde-fou (0.60)
    MAX_PRICE_JUMP              variation simple max du close en 1 s, spot (0.10)
    MAX_PRICE_JUMP_PERP         idem perpétuel (0.15)
    MAX_VWAP_TOL                marge du VWAP hors [low, high] (0.001)
    MAX_PRICE_EXCESS            écart toléré aux extrêmes officiels voisins (0.0005)
    MAX_CLOSE_MISMATCH          part max de minutes au close différent de l'officiel (0.005)
    MAX_VOLUME_DRIFT            dérive max du volume cumulé, part du volume du mois (0.001)
    MAX_DAY_VOLUME_GAP          écart max de volume par jour (0.005)
    MAX_HOUR_VOLUME_GAP         écart max de volume par heure (0.05) ...
    HOUR_VOLUME_FLOOR           ... ou, si plus grand, ce nombre de BTC (100)
    MAX_VOLUME_DEFICIT          écart max du volume du mois, minutes communes (0.01)
    MIN_COMPARED_SHARE          part min des minutes officielles comparées (0.99)
    MAX_EXCLUDED_VOLUME_SHARE   part max du volume officiel exclu par le registre (0.005)
    MAX_FUNDING_ABS / MAX_FUNDING_MEDIAN   bornes du funding rate (0.03 / 0.001)

Le vocabulaire financier (kline, spot, funding, taker…) est défini dans GLOSSAIRE.md.
"""
from __future__ import annotations

import warnings
from datetime import date, timedelta

import pandas as pd
import pytest

import data_contract as contract
import quality_checks as qc
from download_binance import DATASETS
from gaps import (
    SECOND_DATASETS, SHORT_GAP_MAX_SECONDS, STATUSES, approved, data_dir, dataset_gaps,
    detected_long_gaps, files_of, gaps_overlapping, known_gaps_path, known_mismatches_path,
    load_known_gaps, load_known_mismatches, month_of, next_month, split_gaps, utc,
)

DATA = data_dir()


T = qc.THRESHOLDS   # seuils partagés avec diagnose_perp.py (quality_checks.py)
MAX_SHORT_GAP_RATIO = {"spot_klines_1s": T["MAX_SHORT_GAP_RATIO"], "futures_klines_1s": T["MAX_SHORT_GAP_RATIO_PERP"]}
MAX_PRICE_JUMP = {"spot_klines_1s": T["MAX_PRICE_JUMP"], "futures_klines_1s": T["MAX_PRICE_JUMP_PERP"]}
MAX_VWAP_TOL = T["MAX_VWAP_TOL"]
MAX_PRICE_EXCESS = T["MAX_PRICE_EXCESS"]
MAX_CLOSE_MISMATCH = T["MAX_CLOSE_MISMATCH"]
MAX_VOLUME_DRIFT = T["MAX_VOLUME_DRIFT"]
MAX_DAY_VOLUME_GAP = T["MAX_DAY_VOLUME_GAP"]
MAX_HOUR_VOLUME_GAP = T["MAX_HOUR_VOLUME_GAP"]
HOUR_VOLUME_FLOOR = T["HOUR_VOLUME_FLOOR"]
MAX_VOLUME_DEFICIT = T["MAX_VOLUME_DEFICIT"]
MAX_FUNDING_ABS = T["MAX_FUNDING_ABS"]
MAX_FUNDING_MEDIAN = T["MAX_FUNDING_MEDIAN"]

EXPECTED_KLINE_COLS = [c for c in DATASETS["spot_klines_1s"]["columns"] if c != "ignore"]
NUMERIC_KLINE_COLS = [c for c in EXPECTED_KLINE_COLS if c not in ("open_time", "close_time")]

DATASET_MARKS = {
    "spot_klines_1s": [pytest.mark.spot, pytest.mark.klines],
    "futures_klines_1s": [pytest.mark.perp, pytest.mark.klines],
    "futures_klines_1m": [pytest.mark.perp, pytest.mark.reference],
    "futures_funding": [pytest.mark.funding],
}


# --------------------------------------------------------------------------- utilitaires

def dataset_of(path) -> str:
    """Nom du dataset d'un fichier : le nom de son dossier."""
    return path.parent.name


def month_range(path) -> tuple[pd.Timestamp, pd.Timestamp]:
    m = month_of(path)
    return utc(m), utc(next_month(m))


def describe_gaps(gaps: pd.DataFrame, top: int = 10) -> str:
    """Texte lisible listant les trous les plus longs, pour les messages d'erreur."""
    if gaps.empty:
        return ""
    g = gaps.sort_values("duration_s", ascending=False).head(top)
    return "\n".join(f"  {r.start} -> {r.end}  ({r.duration_s} s)" for r in g.itertuples())


def params_for(*datasets: str):
    """Un paramètre pytest par fichier mensuel, étiqueté avec son dataset."""
    params = []
    for dataset in datasets:
        marks = DATASET_MARKS.get(dataset, [])
        params += [pytest.param(p, id=p.stem, marks=marks) for p in files_of(dataset, DATA)]
    if not params:
        return [pytest.param(None, marks=pytest.mark.skip(reason=f"aucun fichier pour {', '.join(datasets)}"))]
    return params


def dataset_params(*datasets: str):
    return [pytest.param(d, id=d, marks=DATASET_MARKS.get(d, [])) for d in datasets]


def assert_no_errors(name: str, errors: list[str]) -> None:
    assert not errors, f"{name}\n" + "\n".join(errors)


# --------------------------------------------------------------------------- fixtures
# scope="module" + params : pytest regroupe les tests par fichier, chaque Parquet n'est lu
# qu'une seule fois même si une vingtaine de tests l'utilisent.

@pytest.fixture(scope="module", params=params_for(*SECOND_DATASETS))
def klines(request):
    """(chemin, DataFrame) d'un fichier mensuel de bougies 1 s (spot ou perpétuel)."""
    return request.param, pd.read_parquet(request.param)


@pytest.fixture(scope="module")
def kline_gaps(klines):
    """
    Trous (courts et longs) qui touchent le mois du fichier, calculés sur tout le dataset
    puis fusionnés aux changements de mois : un trou coupé par minuit en fin de mois est
    classé selon sa durée totale.
    """
    path, _ = klines
    start, end = month_range(path)
    gaps = gaps_overlapping(dataset_gaps(dataset_of(path), str(DATA)), start, end)
    short, long_ = split_gaps(gaps, SHORT_GAP_MAX_SECONDS)
    return short, long_


@pytest.fixture(scope="module")
def known_gaps():
    return load_known_gaps(known_gaps_path())


@pytest.fixture(scope="module")
def known_mismatches():
    return load_known_mismatches(known_mismatches_path())


@pytest.fixture(scope="module", params=params_for("futures_klines_1m"))
def reference_1m(request):
    """(chemin, DataFrame) d'un fichier mensuel de bougies 1 min officielles."""
    return request.param, pd.read_parquet(request.param)


@pytest.fixture(scope="module", params=params_for("futures_funding"))
def funding(request):
    """(chemin, DataFrame) d'un fichier mensuel de funding rate."""
    return request.param, pd.read_parquet(request.param)


# =========================================================================== bougies 1 s : structure

@pytest.mark.structure
class TestKlinesStructure:
    """Le fichier a-t-il la forme attendue ? Prérequis de tous les autres tests."""

    def test_not_empty(self, klines):
        """
        Le fichier contient au moins une ligne.

        Pourquoi : un fichier vide est le symptôme typique d'un téléchargement interrompu
        ou d'un fichier ZIP mal lu. Comme download_binance.py ne re-télécharge pas un mois
        déjà présent sur disque, un fichier vide resterait vide pour toujours sans ce test.
        Un fichier ABSENT n'est pas vu ici : c'est le rôle du contrat de couverture.
        """
        path, df = klines
        assert len(df) > 0, f"{path.name} est vide : relance download_binance.py --force pour ce mois"

    def test_columns(self, klines):
        """
        Les colonnes du fichier stocké sont exactement celles attendues, dans le bon ordre.

        Portée : c'est le schéma de SORTIE du parser qui est vérifié. Les CSV spot de Binance
        n'ont pas d'en-tête et sont lus par position : un décalage des colonnes source
        passerait ce test, mais serait attrapé par les contrôles de valeurs (OHLC, VWAP,
        volume acheteur) et, pour le perpétuel, par la comparaison aux bougies officielles.
        """
        _, df = klines
        assert list(df.columns) == EXPECTED_KLINE_COLS

    def test_dtypes(self, klines):
        """
        Dates en datetime UTC, colonnes numériques, n_trades entier.

        Pourquoi : une colonne de prix lue comme texte casse silencieusement les calculs.
        Mélanger UTC et heure de Paris décale les données d'une ou deux heures, assez pour
        introduire du « futur » dans les features (fuite de données). Un compteur de trades
        non entier signalerait une erreur de parsing.
        """
        _, df = klines
        for col in ("open_time", "close_time"):
            assert pd.api.types.is_datetime64_any_dtype(df[col]), f"{col} n'est pas un datetime"
            assert str(df[col].dt.tz) == "UTC", f"{col} n'est pas en UTC"
        for col in NUMERIC_KLINE_COLS:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} n'est pas numérique"
        assert_no_errors("types", qc.integer_errors(df, "n_trades"))

    def test_finite_values(self, klines):
        """
        Aucune valeur manquante (NaN / NaT) ni infinie.

        Pourquoi : « numérique et non NaN » ne veut pas dire « fini » : +inf passe les
        contrôles de signe et de type. download_binance.py convertit avec errors="coerce",
        donc une valeur illisible devient NaN sans erreur. Beaucoup de modèles refusent NaN
        et inf, et ceux qui les acceptent leur donnent un sens particulier.
        """
        path, df = klines
        errors = qc.finite_errors(df, NUMERIC_KLINE_COLS)
        nat = df[["open_time", "close_time"]].isna().sum()
        if nat.sum():
            errors.append(f"dates manquantes : {nat[nat > 0].to_dict()}")
        assert_no_errors(path.name, errors)


# =========================================================================== bougies 1 s : temps

@pytest.mark.chronologie
class TestKlinesTime:
    """L'axe du temps est-il propre ? C'est l'épine dorsale de toute série temporelle."""

    def test_no_duplicates(self, klines):
        """
        Chaque seconde n'apparaît qu'une seule fois.

        Pourquoi : un doublon décale d'une ligne tous les calculs du type « prix dans 60
        lignes », qui ne correspondent plus à « prix dans 60 secondes ». Portée : ce test
        voit les bougies en double, pas des trades en double déjà agrégés dans une même
        bougie (ceux-là sont traités à la reconstruction et détectés par `verification`).
        """
        _, df = klines
        dup = df["open_time"].duplicated()
        assert not dup.any(), f"{dup.sum()} timestamps en double, ex. {df.loc[dup, 'open_time'].head(3).tolist()}"

    def test_sorted(self, klines):
        """
        Les lignes sont triées par ordre chronologique.

        Pourquoi : les features et les cibles sont calculées avec des décalages de lignes.
        Sur des données mal triées, le modèle peut voir le futur sans que rien ne le signale.
        """
        _, df = klines
        assert df["open_time"].is_monotonic_increasing

    def test_timestamps_in_file_month(self, klines):
        """
        Toutes les bougies appartiennent au mois indiqué par le nom du fichier.

        Pourquoi : le découpage train / test se fera par période. C'est aussi un bon
        détecteur d'erreur d'unité de timestamp (ms au lieu de µs). Portée : ce test ne
        prouve pas que TOUT le mois est présent (rôle des tests de trous).
        """
        path, df = klines
        start, end = month_range(path)
        out = df[(df["open_time"] < start) | (df["open_time"] >= end)]
        assert out.empty, f"{len(out)} lignes hors du mois {start:%Y-%m}, ex. {out['open_time'].head(3).tolist()}"

    def test_aligned_on_second(self, klines):
        """
        Chaque bougie commence pile sur une seconde.

        Pourquoi : la grille temporelle doit être régulière pour que « une ligne = une
        seconde » soit vrai après le remplissage des trous courts.
        """
        _, df = klines
        misaligned = df["open_time"] != df["open_time"].dt.floor("s")
        assert not misaligned.any(), f"{misaligned.sum()} open_time non alignés sur la seconde"

    def test_durations(self, klines):
        """
        Chaque bougie dure une seconde : open_time <= close_time < open_time + 1 s.

        Fin normale : open_time + 999 ms, ou + 999,999 ms pour les timestamps en
        microsecondes du spot depuis 2025.

        Exception, la bougie TRONQUÉE par un arrêt du marché : quand Binance interrompt le
        trading au milieu d'une seconde, la dernière bougie est fermée à l'instant de
        l'arrêt (parfois 0 ms). Elle est acceptée si et seulement si la seconde suivante
        est absente, y compris quand cette seconde appartiendrait au fichier du mois
        suivant. Une durée NÉGATIVE n'est jamais acceptée.

        Pourquoi : une bougie de 3 secondes aurait un volume anormalement grand et
        fausserait les features de volume et de volatilité.
        """
        path, df = klines
        files = files_of(dataset_of(path), DATA)
        i = files.index(path)
        next_first = None
        if i + 1 < len(files):
            next_first = pd.read_parquet(files[i + 1], columns=["open_time"])["open_time"].min()
        assert_no_errors(path.name, qc.duration_errors(df, next_first))


# =========================================================================== bougies 1 s : trous

@pytest.mark.completude
class TestKlinesGaps:
    """
    Les secondes manquantes sont-elles explicables ?

    Deux traitements, selon la durée (voir gaps.py) :
    - courts (<= SHORT_GAP_MAX_SECONDS) : on reportera le dernier prix, volume à 0 ;
    - longs : on coupera la série à cet endroit.
    Les trous sont fusionnés aux changements de mois avant d'être classés.
    """

    def test_short_gaps_are_rare(self, klines, kline_gaps):
        """
        Les trous courts représentent une faible part du mois (MAX_SHORT_GAP_RATIO).

        Pourquoi : si les secondes comblées artificiellement deviennent une part
        significative des données, le modèle apprend surtout du « rien ne se passe »
        fabriqué par nous. Le seuil dépend du marché : 0,5 % sur le spot, 60 % sur le
        perpétuel (qui a réellement beaucoup de secondes sans trade, surtout en 2020 ; sa
        complétude est vérifiée par `verification`). Ce ratio ne prouve pas la CAUSE des
        trous.
        """
        path, _ = klines
        short, _ = kline_gaps
        start, end = month_range(path)
        clipped_start = short["start"].where(short["start"] > start, start)
        clipped_end = short["end"].where(short["end"] < end - qc.ONE_S, end - qc.ONE_S)
        missing = int(((clipped_end - clipped_start) // qc.ONE_S + 1).clip(lower=0).sum())
        n_expected = int((end - start) / qc.ONE_S)
        limit = MAX_SHORT_GAP_RATIO[dataset_of(path)]
        ratio = missing / n_expected
        assert ratio <= limit, (
            f"{path.name} : {missing} s manquantes en trous courts ({ratio:.4%}, max {limit:.4%})\n"
            f"{len(short)} trous courts, les plus longs :\n{describe_gaps(short)}"
        )

    def test_long_gaps_are_approved(self, klines, kline_gaps, known_gaps):
        """
        Chaque trou long est inscrit dans data/known_gaps.csv ET approuvé, avec une raison.

        Pourquoi : un trou long a deux explications possibles qui appellent des réactions
        opposées : fermeture réelle de l'exchange (on coupe la série) ou téléchargement
        raté (on re-télécharge). L'inscription automatique par gaps.py n'est qu'une
        DÉTECTION (statut candidate) : seul un examen humain, tracé par le statut approved
        et une raison, permet d'accepter le trou. Une journée entière manquante n'est
        jamais une maintenance.
        """
        path, _ = klines
        _, long_ = kline_gaps
        ok = approved(known_gaps)
        ok = {(r.start, r.end) for r in ok.itertuples() if r.dataset == dataset_of(path)}
        pending = long_[[(r.start, r.end) not in ok for r in long_.itertuples()]]
        assert pending.empty, (
            f"{path.name} : {len(pending)} trou(s) long(s) non approuvé(s) dans {known_gaps_path()}\n"
            f"{describe_gaps(pending)}\n"
            "Lance python gaps.py, examine chaque trou (maintenance annoncée ? téléchargement raté ?), "
            "puis python approve.py gaps <date> \"<raison>\". Sinon, re-télécharge le mois."
        )


# =========================================================================== bougies 1 s : prix

@pytest.mark.coherence
class TestKlinesPrices:
    """Les prix sont-ils économiquement plausibles ?"""

    def test_prices_positive(self, klines):
        """
        Tous les prix sont strictement positifs (et finis, voir test_finite_values).

        Pourquoi : un prix nul ou négatif casse les rendements logarithmiques (log(0) = -inf).
        """
        _, df = klines
        prices = df[["open", "high", "low", "close"]]
        assert (prices > 0).all().all(), "prix nuls ou négatifs"

    def test_ohlc_consistency(self, klines):
        """
        Le plus haut est au-dessus de l'ouverture et de la clôture, le plus bas en dessous.

        Pourquoi : c'est la définition d'une bougie. Portée : une bougie fausse mais
        cohérente passe ce test ; pour le perpétuel, la comparaison aux bougies officielles
        complète ce contrôle.
        """
        _, df = klines
        bad_high = df["high"] < df[["open", "close"]].max(axis=1)
        bad_low = df["low"] > df[["open", "close"]].min(axis=1)
        assert not bad_high.any(), f"{bad_high.sum()} bougies avec high < max(open, close)"
        assert not bad_low.any(), f"{bad_low.sum()} bougies avec low > min(open, close)"

    def test_no_absurd_price_jump(self, klines):
        """
        Pas de variation SIMPLE du prix de clôture supérieure à MAX_PRICE_JUMP d'une seconde
        à l'autre, dans un sens comme dans l'autre (±10 % sur le spot, ±15 % sur le
        perpétuel).

        Pourquoi : un tel saut indique presque toujours une erreur (autre paire, virgule
        décalée), qui fausserait la normalisation des features. Le perpétuel a un seuil
        plus large : le 18/04/2021 à 03:35:44, il a pris 12 % en une seconde, et la bougie
        1 min officielle confirme ce plus haut. Seules les secondes consécutives sont
        comparées ; ce test porte sur les clôtures, pas sur les mèches intra-seconde.
        """
        path, df = klines
        assert_no_errors(path.name, qc.price_jump_errors(df, MAX_PRICE_JUMP[dataset_of(path)]))


# =========================================================================== bougies 1 s : volumes

@pytest.mark.volumes
class TestKlinesVolumes:
    """Les volumes sont-ils cohérents entre eux et avec les prix ?"""

    def test_volumes(self, klines):
        """
        Volumes positifs, et volume acheteur <= volume total.

        Pourquoi : le déséquilibre acheteurs / vendeurs (taker_buy_base / volume) est une
        des features les plus informatives à court terme. Il doit rester entre 0 et 1.
        """
        _, df = klines
        for col in ("volume", "quote_volume", "taker_buy_base", "taker_buy_quote", "n_trades"):
            assert (df[col] >= 0).all(), f"{col} contient des valeurs négatives"
        assert (df["taker_buy_base"] <= df["volume"] * (1 + 1e-9)).all(), "taker_buy_base > volume"
        assert (df["taker_buy_quote"] <= df["quote_volume"] * (1 + 1e-9)).all(), "taker_buy_quote > quote_volume"

    def test_zero_consistency(self, klines):
        """
        Les volumes nuls vont ensemble : volume nul <=> quote_volume nul (idem acheteur), et
        pas de trades sans volume.

        Pourquoi : un quote_volume sans volume (ou l'inverse) signale des colonnes décalées.
        """
        path, df = klines
        assert_no_errors(path.name, qc.zero_consistency_errors(df))

    def test_vwap_in_range(self, klines):
        """
        Les prix moyens implicites restent dans [low, high], à MAX_VWAP_TOL près (0,1 %) :
        quote_volume / volume pour tous les trades, taker_buy_quote / taker_buy_base pour
        les trades à l'initiative de l'acheteur.

        Pourquoi : c'est un contrôle croisé fort entre les colonnes de prix et de volume.
        La marge de 0,1 % est une marge de prudence pour les arrondis des volumes publiés ;
        elle n'a pas été calibrée sur des cas réels.
        """
        path, df = klines
        assert_no_errors(path.name, qc.vwap_errors(df, MAX_VWAP_TOL))


# =========================================================================== référence 1 min officielle

class TestReference1m:
    """
    Les bougies 1 min officielles servent d'ORACLE pour vérifier le perpétuel : elles
    reçoivent donc leurs propres contrôles de structure, de temps et de valeurs.
    """

    @pytest.mark.structure
    def test_structure(self, reference_1m):
        """Fichier non vide, colonnes attendues, dates UTC, valeurs finies, n_trades entier."""
        path, df = reference_1m
        assert len(df) > 0, f"{path.name} est vide"
        assert list(df.columns) == EXPECTED_KLINE_COLS
        errors = qc.finite_errors(df, NUMERIC_KLINE_COLS) + qc.integer_errors(df, "n_trades")
        for col in ("open_time", "close_time"):
            if str(df[col].dt.tz) != "UTC":
                errors.append(f"{col} n'est pas en UTC")
        assert_no_errors(path.name, errors)

    @pytest.mark.chronologie
    def test_time_axis(self, reference_1m):
        """Pas de doublon, trié, dans le mois du fichier, aligné sur la minute, durée d'une minute."""
        path, df = reference_1m
        start, end = month_range(path)
        errors = []
        if df["open_time"].duplicated().any():
            errors.append(f"{df['open_time'].duplicated().sum()} minutes en double")
        if not df["open_time"].is_monotonic_increasing:
            errors.append("minutes non triées")
        outside = (df["open_time"] < start) | (df["open_time"] >= end)
        if outside.any():
            errors.append(f"{outside.sum()} minutes hors du mois")
        if (df["open_time"] != df["open_time"].dt.floor("min")).any():
            errors.append("open_time non alignés sur la minute")
        delta = df["close_time"] - df["open_time"]
        bad = (delta < pd.Timedelta(seconds=59)) | (delta >= qc.ONE_MIN)
        if bad.any():
            errors.append(f"{bad.sum()} bougies dont la durée n'est pas d'une minute")
        assert_no_errors(path.name, errors)

    @pytest.mark.coherence
    def test_values(self, reference_1m):
        """Prix positifs, OHLC cohérents, volumes positifs, acheteur <= total, nuls cohérents."""
        path, df = reference_1m
        errors = []
        if not (df[["open", "high", "low", "close"]] > 0).all().all():
            errors.append("prix nuls ou négatifs")
        if (df["high"] < df[["open", "close"]].max(axis=1)).any() or (df["low"] > df[["open", "close"]].min(axis=1)).any():
            errors.append("OHLC incohérents")
        if (df[["volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]] < 0).any().any():
            errors.append("volumes négatifs")
        if (df["taker_buy_base"] > df["volume"] * (1 + 1e-9)).any():
            errors.append("taker_buy_base > volume")
        errors += [e for e in qc.zero_consistency_errors(df) if "trades mais" not in e]
        assert_no_errors(path.name, errors)


# =========================================================================== contrat de couverture

@pytest.mark.completude
class TestCoverageContract:
    """
    Les données présentes sont-elles celles attendues (data_contract.py) ?

    Pourquoi : les autres tests ne voient que les fichiers présents. Sans ce contrat, un
    téléchargement partiel ou un mauvais dossier de données réduit silencieusement ce qui
    est contrôlé, et pytest peut rester vert avec des dizaines de tests ignorés. En mode
    VALIDATION_MODE=partial, ces tests sont ignorés avec un avertissement : un résultat
    partiel ne vaut pas validation.
    """

    @pytest.fixture(autouse=True)
    def _mode(self):
        if contract.validation_mode() == "partial":
            warnings.warn("VALIDATION_MODE=partial : contrat de couverture NON vérifié", stacklevel=1)
            pytest.skip("mode partial : contrat de couverture non vérifié")

    @pytest.mark.parametrize("dataset", dataset_params(*contract.REQUIRED_DATASETS))
    def test_dataset_present_from_expected_start(self, dataset):
        """Le dataset existe et commence au premier mois attendu (janvier 2020)."""
        files = files_of(dataset, DATA)
        assert files, f"{dataset} : aucun fichier dans {DATA / dataset}"
        first = f"{month_of(files[0]):%Y-%m}"
        assert first == contract.REQUIRED_DATASETS[dataset], (
            f"{dataset} : premier mois {first}, attendu {contract.REQUIRED_DATASETS[dataset]}")

    @pytest.mark.parametrize("dataset", dataset_params(*contract.REQUIRED_DATASETS))
    def test_up_to_expected_last_month(self, dataset):
        """
        Le dernier mois présent est au moins le dernier mois attendu (fraîcheur) : le mois
        en cours pour les datasets journaliers, le mois précédent pour le funding (publié
        mensuellement).
        """
        files = files_of(dataset, DATA)
        assert files, f"{dataset} : aucun fichier"
        last, expected = month_of(files[-1]), contract.expected_last_month(dataset)
        assert last >= expected, f"{dataset} : dernier mois {last:%Y-%m}, attendu au moins {expected:%Y-%m}"

    @pytest.mark.parametrize("dataset", dataset_params(*sorted(contract.DAILY_DATASETS)))
    def test_recent_days_complete(self, dataset):
        """
        Pour les datasets journaliers, la journée d'avant-hier est présente jusqu'à sa
        dernière minute (Binance publie les fichiers journaliers avec un jour de décalage).
        """
        files = files_of(dataset, DATA)
        assert files, f"{dataset} : aucun fichier"
        last = pd.read_parquet(files[-1], columns=["open_time"])["open_time"].max()
        limit = utc(date.today() - timedelta(days=1)) - pd.Timedelta(minutes=2)
        assert last >= limit, f"{dataset} : dernière donnée au {last}, attendue après {limit}. Relance download_binance.py"

    @pytest.mark.parametrize("dataset", dataset_params(*contract.REQUIRED_DATASETS))
    def test_no_missing_month(self, dataset):
        """
        Aucun mois manquant entre le premier et le dernier fichier.

        Pourquoi : download_binance.py ignore un mois indisponible (404). Un mois manquant
        au milieu de l'historique n'a pas de fichier à tester.
        """
        files = files_of(dataset, DATA)
        assert files, f"{dataset} : aucun fichier"
        present = {month_of(p) for p in files}
        m, last = min(present), max(present)
        missing = []
        while m <= last:
            if m not in present:
                missing.append(f"{m:%Y-%m}")
            m = next_month(m)
        assert not missing, f"{dataset} : mois manquants {missing}"

    @pytest.mark.perp
    @pytest.mark.verification
    def test_reference_for_every_perp_month(self):
        """Chaque mois du perpétuel 1 s a sa référence 1 min officielle (sinon il n'est pas vérifié)."""
        perp = {month_of(p) for p in files_of("futures_klines_1s", DATA)}
        ref = {month_of(p) for p in files_of("futures_klines_1m", DATA)}
        missing = sorted(perp - ref)
        assert perp, "aucun fichier futures_klines_1s"
        assert not missing, f"mois du perpétuel sans référence 1 min : {[f'{m:%Y-%m}' for m in missing]}"


@pytest.mark.structure
def test_no_leftover_tmp_files():
    """
    Aucun fichier .tmp ne traîne dans le dossier de données.

    Pourquoi : download_binance.py écrit d'abord dans un .tmp puis le renomme. Un .tmp
    restant signifie qu'une écriture a été interrompue.
    """
    tmp = list(DATA.rglob("*.tmp"))
    assert not tmp, f"fichiers temporaires restants : {tmp}"


# =========================================================================== registres

@pytest.mark.registre
class TestGapRegistry:
    """Le registre des trous longs est-il bien formé, examiné et à jour ?"""

    def test_well_formed(self, known_gaps):
        """
        Datasets connus, bornes alignées sur la seconde, fin après début, durée cohérente,
        statut valide, raison obligatoire pour une entrée approuvée.

        Pourquoi : le registre est édité à la main et servira à couper les séries lors de
        la construction des features. Une ligne corrompue ferait couper au mauvais endroit.
        """
        if known_gaps.empty:
            pytest.skip("registre vide ou absent, lance `python gaps.py`")
        g = known_gaps
        errors = []
        if not g["dataset"].isin(SECOND_DATASETS).all():
            errors.append(f"dataset inconnu : {sorted(set(g['dataset']) - set(SECOND_DATASETS))}")
        if ((g["start"] != g["start"].dt.floor("s")) | (g["end"] != g["end"].dt.floor("s"))).any():
            errors.append("bornes non alignées sur la seconde")
        if (g["end"] < g["start"]).any():
            errors.append("trou avec end < start")
        computed = ((g["end"] - g["start"]) // qc.ONE_S + 1).astype("int64")
        if (computed != g["duration_s"]).any():
            errors.append("duration_s incohérent avec start / end")
        if not g["status"].isin(STATUSES).all():
            errors.append(f"statut invalide : {sorted(set(g['status']) - set(STATUSES))}")
        if ((g["status"] == "approved") & (g["reason"].str.strip() == "")).any():
            errors.append("entrée approuvée sans raison")
        if g.duplicated(["dataset", "start", "end"]).any():
            errors.append("entrées en double")
        assert_no_errors(str(known_gaps_path()), errors)

    def test_no_overlap(self, known_gaps):
        """Les trous du registre ne se chevauchent pas (même dataset)."""
        if known_gaps.empty:
            pytest.skip("registre vide ou absent, lance `python gaps.py`")
        for dataset, g in known_gaps.groupby("dataset"):
            g = g.sort_values("start")
            overlap = g["start"].iloc[1:].values <= g["end"].iloc[:-1].values
            assert not overlap.any(), f"{dataset} : {overlap.sum()} trous qui se chevauchent"

    def test_no_stale_entry(self, known_gaps):
        """
        Chaque entrée du registre correspond encore à un trou long détecté.

        Pourquoi : après une réparation, un trou disparaît ; son exception devient
        obsolète et pourrait masquer un futur problème au même endroit.
        """
        if known_gaps.empty:
            pytest.skip("registre vide ou absent")
        detected = detected_long_gaps()
        keys = set(zip(detected["dataset"], detected["start"], detected["end"]))
        stale = known_gaps[[(r.dataset, r.start, r.end) not in keys for r in known_gaps.itertuples()]]
        assert stale.empty, f"{len(stale)} entrée(s) obsolète(s) :\n{stale.head(10).to_string(index=False)}\nRelance python gaps.py."


@pytest.mark.registre
@pytest.mark.perp
class TestMismatchRegistry:
    """Le registre des minutes irréparables du perpétuel est-il bien formé ?"""

    def test_well_formed(self, known_mismatches):
        """
        Types de minute connus, minutes alignées, pas de doublon ni de minute inscrite sous
        deux types, statut valide, raison obligatoire pour une entrée approuvée.
        """
        if known_mismatches.empty:
            pytest.skip("registre vide ou absent")
        assert_no_errors(str(known_mismatches_path()), qc.registry_errors(known_mismatches))


# =========================================================================== vérification croisée

def perp_month_params():
    params = []
    ref = {month_of(p): p for p in files_of("futures_klines_1m", DATA)}
    for p in files_of("futures_klines_1s", DATA):
        m = month_of(p)
        marks = [pytest.mark.skip(reason="référence 1 min absente (voir le contrat de couverture)")] if m not in ref else []
        params.append(pytest.param((p, ref.get(m)), id=f"{m:%Y-%m}", marks=marks))
    return params or [pytest.param(None, marks=pytest.mark.skip(reason="aucun fichier futures_klines_1s"))]


@pytest.fixture(scope="module", params=perp_month_params())
def perp_pair(request):
    """
    Pour un mois : nos bougies 1 s agrégées en 1 min, et les bougies 1 min officielles
    (minutes avec trades), après application des exceptions APPROUVÉES du registre.
    """
    path_1s, path_1m = request.param
    start, end = month_range(path_1s)
    ours_full = qc.to_minutes(pd.read_parquet(path_1s))
    official_full = qc.official_minutes(pd.read_parquet(path_1m))
    ours, official, reg_errors, stats = qc.apply_registry(ours_full, official_full, load_known_mismatches(), start, end)
    return {"name": path_1s.name, "ours": ours, "official": official, "ours_full": ours_full,
            "registry_errors": reg_errors, "stats": stats}


@pytest.mark.perp
@pytest.mark.verification
class TestPerpetualVsOfficial1m:
    """
    Les bougies 1 s du perpétuel, reconstruites à partir des aggTrades, concordent-elles
    avec les bougies 1 min officielles de Binance ?

    C'est un contrôle par un pipeline de calcul indépendant, mais du MÊME fournisseur :
    des lacunes communes aux deux produits ne seraient pas vues. Même une concordance
    parfaite à la minute ne prouverait pas l'exactitude seconde par seconde ; la logique
    de reconstruction elle-même est vérifiée par les tests unitaires.

    Écarts attendus et leur traitement :
    - des trades proches d'un changement de minute sont parfois rangés dans la minute
      voisine chez Binance (76 % des minutes en excès compensées exactement par la voisine
      en juin 2023, volume du mois identique). Selon la documentation de Binance, les
      aggTrades regroupent les trades de même prix et même côté sur 100 ms et excluent
      ceux du fonds d'assurance et de l'ADL : cela rend plausible une différence de
      granularité et de périmètre, sans prouver le mécanisme de chaque écart. L'effet d'un
      tel déplacement sur des features à la seconde n'est PAS démontré négligeable ;
    - n_trades n'est pas comparé : notre valeur, calculée à partir des plages
      d'identifiants, dépasse l'officielle même quand les volumes concordent. C'est une
      approximation, à ne pas utiliser comme un nombre exact.
    """

    def test_registry_up_to_date(self, perp_pair):
        """
        Les exceptions du mois sont toutes examinées (pas de candidate) et aucune n'est
        obsolète (une minute déclarée absente est désormais présente).
        """
        assert_no_errors(perp_pair["name"], perp_pair["registry_errors"])

    def test_comparison_coverage(self, perp_pair):
        """
        Assez de minutes sont réellement comparées : au moins MIN_COMPARED_SHARE (99 %) des
        minutes officielles, et le registre n'exclut pas plus de MAX_EXCLUDED_VOLUME_SHARE
        (0,5 %) du volume officiel du mois.

        Pourquoi : chaque exclusion réduit ce qui est vérifié. Sans cette borne, un
        registre trop généreux pourrait vider la vérification de son sens.
        """
        s = perp_pair["stats"]
        errors = []
        if s["compared_share"] < contract.MIN_COMPARED_SHARE:
            errors.append(f"{s['compared_minutes']} minutes comparées sur {s['official_minutes']} "
                          f"({s['compared_share']:.2%}, min {contract.MIN_COMPARED_SHARE:.0%})")
        if s["excluded_official_volume_share"] > contract.MAX_EXCLUDED_VOLUME_SHARE:
            errors.append(f"volume officiel exclu {s['excluded_official_volume_share']:.3%} "
                          f"(max {contract.MAX_EXCLUDED_VOLUME_SHARE:.2%})")
        assert_no_errors(perp_pair["name"], errors)

    def test_no_missing_minutes(self, perp_pair):
        """
        Mêmes minutes avec des trades des deux côtés, après exceptions approuvées.

        Une minute officielle absente chez nous signifie des trades perdus ; des blocs de
        1 440 minutes indiquent un fichier mensuel incomplet (download_binance.py le répare
        avec les fichiers journaliers). Une minute à nous absente de l'officiel signifie
        une référence incomplète : elle n'est plus un simple avertissement, elle doit être
        examinée et approuvée (`python diagnose_perp.py --register`, puis approve.py).
        """
        missing, extra = qc.missing_minutes(perp_pair["ours"], perp_pair["official"])
        errors = []
        if len(missing):
            days = pd.Series(missing.floor("D")).value_counts()
            full = [f"{d:%Y-%m-%d}" for d, n in days.items() if n >= 1440]
            errors.append(f"{len(missing)} minute(s) officielle(s) absente(s) de nos données, ex. "
                          f"{list(missing[:3])} ; journées entières : {full or 'aucune'}")
        if len(extra):
            errors.append(f"{len(extra)} minute(s) absente(s) des bougies officielles, ex. {list(extra[:3])}")
        assert_no_errors(perp_pair["name"], errors)

    def test_prices_match_neighbours(self, perp_pair):
        """
        Extrêmes cohérents avec les minutes officielles EXACTES t - 1, t, t + 1, à
        MAX_PRICE_EXCESS près (0,05 %), dans les deux sens : ni extrême inventé (high trop
        haut, low trop bas), ni mèche disparue (extrême officiel absent de nos données).

        Pourquoi : les simulations de stop-loss du backtest se déclenchent sur les extrêmes.
        Une mèche perdue rend le backtest optimiste ; un extrême inventé déclenche des
        stops fictifs.
        """
        errors = qc.price_errors(perp_pair["ours"], perp_pair["official"], MAX_PRICE_EXCESS, perp_pair["ours_full"])
        assert_no_errors(perp_pair["name"], errors)

    def test_close_matches(self, perp_pair):
        """
        Le close de chaque minute est identique à l'officiel, sauf pour au plus
        MAX_CLOSE_MISMATCH (0,5 %) des minutes.

        Pourquoi : l'égalité des close a été constatée sur 100 % des minutes des mois
        diagnostiqués ; elle garantit que le dernier trade de chaque minute est bien le
        bon. L'open n'est PAS contrôlé : il diffère légitimement quand un trade de début de
        minute est rangé dans la minute précédente.
        """
        share = qc.close_mismatch_share(perp_pair["ours"], perp_pair["official"])
        assert share <= MAX_CLOSE_MISMATCH, (
            f"{perp_pair['name']} : close différent sur {share:.3%} des minutes (max {MAX_CLOSE_MISMATCH:.2%})")

    def test_no_volume_drift(self, perp_pair):
        """
        L'écart de volume cumulé (volume et volume acheteur) ne dépasse jamais
        MAX_VOLUME_DRIFT (0,1 %) du volume du mois.

        Un trade rangé dans la minute voisine se compense aussitôt ; un trade perdu, compté
        deux fois ou au mauvais sens crée un écart qui persiste. C'est ce test qui a révélé
        les aggTrades en double de septembre 2022. Portée : une borne d'amplitude sur le
        mois ; les erreurs locales sont contrôlées par test_local_volumes.
        """
        errors = qc.drift_errors(perp_pair["ours"], perp_pair["official"], MAX_VOLUME_DRIFT)
        assert_no_errors(perp_pair["name"], errors)

    def test_local_volumes(self, perp_pair):
        """
        Volumes (et volume acheteur) concordants par JOUR (MAX_DAY_VOLUME_GAP, 0,5 %) et par
        HEURE (MAX_HOUR_VOLUME_GAP, 5 %, ou HOUR_VOLUME_FLOOR BTC si c'est plus grand).

        Pourquoi : une heure amputée de 30 % ne pèse que 0,04 % d'un mois et passe la
        dérive mensuelle. Le plancher horaire en BTC absorbe les trades déplacés à la
        frontière d'une heure (jusqu'à 60 BTC observés).
        """
        errors = qc.local_volume_errors(perp_pair["ours"], perp_pair["official"],
                                        MAX_DAY_VOLUME_GAP, MAX_HOUR_VOLUME_GAP, HOUR_VOLUME_FLOOR)
        assert_no_errors(perp_pair["name"], errors)

    def test_month_volume(self, perp_pair):
        """
        Le volume total des MINUTES COMMUNES concorde à MAX_VOLUME_DEFICIT près (1 %), dans
        les deux sens : trop faible = trades perdus, trop élevé = trades en double. Les
        minutes présentes d'un seul côté sont traitées par test_no_missing_minutes.
        """
        gap = qc.month_volume_gap(perp_pair["ours"], perp_pair["official"])
        assert gap <= MAX_VOLUME_DEFICIT, f"{perp_pair['name']} : écart de volume {gap:.3%} (max {MAX_VOLUME_DEFICIT:.2%})"


# =========================================================================== funding

@pytest.mark.funding
class TestFunding:
    """Le funding rate est un coût direct des positions à levier : il doit être fiable."""

    @pytest.mark.structure
    def test_schema(self, funding):
        """
        Colonnes attendues, calc_time en datetime UTC, valeurs finies, intervalle positif.

        Pourquoi : un taux manquant serait probablement traité comme 0 par le backtest,
        c'est-à-dire un paiement gratuit qui n'a pas existé.
        """
        path, df = funding
        errors = []
        if list(df.columns) != DATASETS["futures_funding"]["columns"]:
            errors.append(f"colonnes inattendues : {list(df.columns)}")
        if not pd.api.types.is_datetime64_any_dtype(df["calc_time"]) or str(df["calc_time"].dt.tz) != "UTC":
            errors.append("calc_time n'est pas un datetime UTC")
        errors += qc.finite_errors(df, ["funding_interval_hours", "last_funding_rate"])
        if (df["funding_interval_hours"] <= 0).any():
            errors.append("intervalle déclaré nul ou négatif")
        assert_no_errors(path.name, errors)

    @pytest.mark.chronologie
    def test_no_duplicates_and_sorted(self, funding):
        """Chaque paiement n'apparaît qu'une fois, dans l'ordre chronologique."""
        _, df = funding
        assert not df["calc_time"].duplicated().any()
        assert df["calc_time"].is_monotonic_increasing

    @pytest.mark.chronologie
    def test_timestamps_in_file_month(self, funding):
        """Tous les paiements appartiennent au mois du nom de fichier."""
        path, df = funding
        start, end = month_range(path)
        assert df["calc_time"].between(start, end, inclusive="left").all()

    @pytest.mark.completude
    def test_schedule(self, funding):
        """
        Les paiements sont EXACTEMENT les échéances attendues du mois : un paiement à chaque
        multiple de l'intervalle déclaré (8 h pour BTCUSDT) depuis le début du mois.

        Pourquoi : un paiement manquant (au début, à la fin ou au milieu du mois) ferait
        comme si une position tenue à ce moment-là n'avait rien payé ; un paiement en trop
        la ferait payer deux fois. Comme chaque mois commence à 00:00 et est couvert jusqu'à
        sa fin, la continuité entre deux mois est garantie. Un fichier réduit à une seule
        ligne échoue.
        """
        path, df = funding
        start, end = month_range(path)
        assert_no_errors(path.name, qc.funding_errors(df, start, end))

    @pytest.mark.coherence
    def test_rate_magnitude(self, funding):
        """
        |taux| < MAX_FUNDING_ABS (3 %) sur chaque paiement, et médiane de |taux| <=
        MAX_FUNDING_MEDIAN (0,1 %) sur le mois.

        Pourquoi : un taux typique est de l'ordre de 0,01 %. La borne maximale seule
        laisserait passer une erreur d'unité d'un facteur 100 (0,01 % lu comme 1 %) ; la
        médiane l'attrape.
        """
        path, df = funding
        assert_no_errors(path.name, qc.funding_rate_errors(df, MAX_FUNDING_ABS, MAX_FUNDING_MEDIAN))