"""
Tests de qualité des données téléchargées par download_binance.py.

Pourquoi tester des données ?
-----------------------------
Un modèle de trading n'est jamais meilleur que les données sur lesquelles il apprend.
Une erreur dans les données ne fait pas planter le code : elle produit silencieusement
un modèle qui semble excellent en backtest et qui perd de l'argent en réel. Par exemple :

- une seconde en double fait apparaître un rendement nul qui n'a pas existé ;
- un prix aberrant crée un « signal » que le modèle va apprendre par cœur ;
- un trou de 3 h comblé naïvement fait croire à un marché immobile ;
- une bougie rangée dans le mauvais mois fausse le découpage train / test.

Ces tests sont donc la première barrière entre les fichiers bruts et les modèles.
À relancer après chaque téléchargement.

Lancement (depuis la racine du projet)
--------------------------------------
    pytest                          # tous les tests
    pytest -k gaps                  # seulement les tests sur les trous
    pytest -k "2020-06"             # seulement un mois
    pytest -x -q                    # s'arrête au premier échec

Avant le premier lancement, générer le registre des trous longs :
    python gaps.py

Lancer par catégorie (option -m)
--------------------------------
Chaque test porte une étiquette de catégorie et une étiquette de dataset. `-m` sélectionne
les tests par étiquette, `-k` par nom de test ou de fichier. Les deux se combinent.

    Catégorie       Ce qu'elle vérifie                                    Quand la relancer
    -------------   ---------------------------------------------------   ------------------------------------------
    structure       fichier non vide, colonnes, types, pas de NaN,        après une modification de download_binance.py
                    fichiers .tmp restants                                 ou un changement de format chez Binance
    chronologie     doublons, tri, bon mois, alignement, close_time       après une modification du parsing des dates
                                                                           ou de la fusion (merge_into)
    completude      trous courts / longs, mois manquants, fraîcheur,      après un (re)téléchargement ou après
                    régularité du funding, registre                        `python gaps.py`
    coherence       prix positifs, OHLC, sauts de prix,                   après un (re)téléchargement de fichiers
                    bornes du funding
    volumes         volumes positifs, volume acheteur <= total,           après un (re)téléchargement de fichiers
                    trades => volume, VWAP entre low et high
    registre        known_gaps.csv bien formé (aussi dans completude)     après avoir édité known_gaps.csv à la main
    verification    bougies 1 s du perpétuel agrégées en 1 min =          après une modification de la reconstruction
                    bougies 1 min officielles de Binance                   (build_bars_from_aggtrades) ou un
                                                                           (re)téléchargement du perpétuel

    Dataset         klines   bougies 1 s (spot et perpétuel)
                    spot     bougies 1 s du spot uniquement
                    perp     bougies 1 s du perpétuel uniquement (et leur vérification)
                    funding  funding rate du perpétuel

Exemples :
    pytest -m completude                        # une seule catégorie
    pytest -m "completude or chronologie"       # plusieurs catégories
    pytest -m "not completude"                  # tout sauf une catégorie
    pytest -m funding                           # un seul dataset
    pytest -m "klines and coherence"            # une catégorie sur un dataset
    pytest -m "perp and completude"             # une catégorie sur le perpétuel seulement
    pytest -m completude -k "2021-04"           # une catégorie sur un seul mois
    pytest --lf                                 # seulement les tests qui ont échoué la dernière fois
    pytest --markers                            # liste des étiquettes disponibles

Scénarios courants :
    - Un mois a été re-téléchargé      -> pytest -k "2021-04"
    - Le registre a été régénéré       -> pytest -m "completude or registre"
    - known_gaps.csv édité à la main   -> pytest -m registre
    - Le parsing a été modifié         -> pytest -m "structure or chronologie"
    - La reconstruction 1 s a changé   -> pytest -m verification
    - Vérification complète            -> pytest

Seuls les fichiers Parquet utilisés par les tests sélectionnés sont lus : une sélection
étroite est donc aussi beaucoup plus rapide.

Variables d'environnement optionnelles
--------------------------------------
    BINANCE_DATA_DIR            dossier des données (défaut : data/raw)
    KNOWN_GAPS_FILE             registre des trous longs (défaut : data/known_gaps.csv)
    SHORT_GAP_MAX_SECONDS       durée max d'un trou « court » (défaut : 60)
    MAX_SHORT_GAP_RATIO         part max de secondes manquantes en trous courts (défaut : 0.005)
    MAX_PRICE_JUMP              variation de prix max en 1 s (défaut : 0.10 = 10 %)

Le vocabulaire financier (kline, spot, funding, taker…) est défini dans GLOSSAIRE.md.
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from download_binance import DATASETS
from gaps import (
    SECOND_DATASETS, SHORT_GAP_MAX_SECONDS, data_dir, expected_range, files_of, find_gaps,
    known_gaps_path, load_known_gaps, month_of, next_month, split_gaps, utc,
)

DATA = data_dir()
MAX_SHORT_GAP_RATIO = float(os.environ.get("MAX_SHORT_GAP_RATIO", "0.005"))
MAX_PRICE_JUMP = float(os.environ.get("MAX_PRICE_JUMP", "0.10"))

EXPECTED_KLINE_COLS = [c for c in DATASETS["spot_klines_1s"]["columns"] if c != "ignore"]

# étiquette de marché ajoutée à chaque fichier : `pytest -m spot` ou `pytest -m perp`
MARKET_MARK = {"spot_klines_1s": pytest.mark.spot, "futures_klines_1s": pytest.mark.perp}


# --------------------------------------------------------------------------- utilitaires

def describe_gaps(gaps: pd.DataFrame, top: int = 10) -> str:
    """Texte lisible listant les trous les plus longs, pour les messages d'erreur."""
    if gaps.empty:
        return ""
    g = gaps.sort_values("duration_s", ascending=False).head(top)
    return "\n".join(f"  {r.start} -> {r.end}  ({r.duration_s} s)" for r in g.itertuples())


def params_for(*datasets: str):
    """Un paramètre pytest par fichier mensuel, étiqueté avec son marché (spot / perp)."""
    params = []
    for dataset in datasets:
        mark = MARKET_MARK.get(dataset)
        marks = [mark] if mark else []
        params += [pytest.param(p, id=p.stem, marks=marks) for p in files_of(dataset, DATA)]
    if not params:
        return [pytest.param(None, marks=pytest.mark.skip(reason=f"aucun fichier pour {', '.join(datasets)}"))]
    return params


def dataset_of(path) -> str:
    """Nom du dataset d'un fichier : le nom de son dossier."""
    return path.parent.name


def months_in_both(a: str, b: str):
    """Mois présents à la fois dans les datasets a et b (pour les comparaisons croisées)."""
    fa = {month_of(p): p for p in files_of(a, DATA)}
    fb = {month_of(p): p for p in files_of(b, DATA)}
    common = sorted(fa.keys() & fb.keys())
    if not common:
        return [pytest.param(None, marks=pytest.mark.skip(reason=f"aucun mois commun entre {a} et {b}"))]
    return [pytest.param((fa[m], fb[m]), id=f"{m:%Y-%m}") for m in common]


# --------------------------------------------------------------------------- fixtures
# scope="module" + params : pytest regroupe les tests par fichier, chaque Parquet n'est lu
# qu'une seule fois même si une vingtaine de tests l'utilisent.

@pytest.fixture(scope="module", params=params_for(*SECOND_DATASETS))
def klines(request):
    """(chemin, DataFrame) d'un fichier mensuel de klines 1s (spot ou perpétuel)."""
    return request.param, pd.read_parquet(request.param)


@pytest.fixture(scope="module")
def kline_gaps(klines):
    """Trous (courts et longs) du fichier courant, calculés une seule fois par fichier."""
    path, df = klines
    first = files_of(dataset_of(path), DATA)[0]
    start, end = expected_range(path, df, is_first_file=(path == first))
    gaps = find_gaps(df["open_time"], start, end)
    short, long_ = split_gaps(gaps, SHORT_GAP_MAX_SECONDS)
    n_expected = int((end - start) / pd.Timedelta(seconds=1))
    return short, long_, n_expected


@pytest.fixture(scope="module")
def known_gaps():
    return load_known_gaps(known_gaps_path())


@pytest.fixture(scope="module", params=params_for("futures_funding"))
def funding(request):
    """(chemin, DataFrame) d'un fichier mensuel de funding rate."""
    return request.param, pd.read_parquet(request.param)


# =========================================================================== klines 1s : structure

@pytest.mark.klines
@pytest.mark.structure
class TestKlinesStructure:
    """Le fichier a-t-il la forme attendue ? Prérequis de tous les autres tests."""

    def test_not_empty(self, klines):
        """
        Le fichier contient au moins une ligne.

        Pourquoi : un fichier vide est le symptôme typique d'un téléchargement interrompu
        ou d'un fichier ZIP mal lu. Comme download_binance.py ne re-télécharge pas un mois
        déjà présent sur disque, un fichier vide resterait vide pour toujours sans ce test.
        """
        path, df = klines
        assert len(df) > 0, f"{path.name} est vide : supprime-le et relance download_binance.py"

    def test_columns(self, klines):
        """
        Les colonnes sont exactement celles attendues, dans le bon ordre.

        Pourquoi : Binance nomme les colonnes par position, pas par nom (les CSV spot n'ont
        pas d'en-tête). Si Binance ajoutait ou retirait une colonne, toutes les suivantes
        seraient décalées : le volume deviendrait le prix de clôture, etc. Le code
        continuerait de tourner et le modèle apprendrait n'importe quoi.
        """
        _, df = klines
        assert list(df.columns) == EXPECTED_KLINE_COLS

    def test_dtypes(self, klines):
        """
        Les dates sont des datetimes UTC et toutes les autres colonnes sont numériques.

        Pourquoi : une colonne de prix lue comme texte ne plante pas toujours (pandas sait
        concaténer des chaînes), mais elle casse silencieusement les calculs. Quant au
        fuseau horaire, mélanger UTC et heure de Paris décale les données d'une ou deux
        heures, ce qui suffit à introduire du « futur » dans les features (fuite de données).
        """
        _, df = klines
        for col in ("open_time", "close_time"):
            assert pd.api.types.is_datetime64_any_dtype(df[col]), f"{col} n'est pas un datetime"
            assert str(df[col].dt.tz) == "UTC", f"{col} n'est pas en UTC"
        for col in EXPECTED_KLINE_COLS:
            if col not in ("open_time", "close_time"):
                assert pd.api.types.is_numeric_dtype(df[col]), f"{col} n'est pas numérique"

    def test_no_nan(self, klines):
        """
        Aucune valeur manquante (NaN).

        Pourquoi : download_binance.py convertit les colonnes avec errors="coerce", donc
        une valeur illisible devient NaN au lieu de provoquer une erreur. Beaucoup de
        modèles refusent les NaN, et ceux qui les acceptent (LightGBM, XGBoost) leur
        donnent un sens particulier qui fausserait l'apprentissage.
        """
        _, df = klines
        nan = df.isna().sum()
        assert nan.sum() == 0, f"valeurs manquantes :\n{nan[nan > 0]}"


# =========================================================================== klines 1s : temps

@pytest.mark.klines
@pytest.mark.chronologie
class TestKlinesTime:
    """L'axe du temps est-il propre ? C'est l'épine dorsale de toute série temporelle."""

    def test_no_duplicates(self, klines):
        """
        Chaque seconde n'apparaît qu'une seule fois.

        Pourquoi : un doublon crée un rendement nul entre deux lignes identiques, et
        fait compter deux fois le volume de cette seconde. Surtout, il décale d'une ligne
        tous les calculs du type « prix dans 60 lignes », qui ne correspondent alors plus
        à « prix dans 60 secondes ». Les variables cibles seraient fausses.
        """
        _, df = klines
        dup = df["open_time"].duplicated()
        assert not dup.any(), f"{dup.sum()} timestamps en double, ex. {df.loc[dup, 'open_time'].head(3).tolist()}"

    def test_sorted(self, klines):
        """
        Les lignes sont triées par ordre chronologique.

        Pourquoi : les features (moyennes mobiles, rendements) et les cibles (« le prix
        va-t-il monter ? ») sont calculées avec des décalages de lignes. Sur des données
        mal triées, « la ligne suivante » n'est plus « la seconde suivante », et le modèle
        peut voir le futur sans que rien ne le signale.
        """
        _, df = klines
        assert df["open_time"].is_monotonic_increasing

    def test_timestamps_in_file_month(self, klines):
        """
        Toutes les bougies appartiennent au mois indiqué par le nom du fichier.

        Pourquoi : tu découperas probablement les données en train / validation / test
        par période (ex. 2020-2023 pour entraîner, 2024 pour tester). Une bougie de
        janvier rangée dans le fichier de février brouillerait cette frontière. C'est
        aussi un bon détecteur d'erreur d'unité de timestamp (ms au lieu de µs), qui
        enverrait les dates en 1970 ou en l'an 50 000.
        """
        path, df = klines
        start = utc(month_of(path))
        end = utc(next_month(month_of(path)))
        out = df[(df["open_time"] < start) | (df["open_time"] >= end)]
        assert out.empty, f"{len(out)} lignes hors du mois {start:%Y-%m}, ex. {out['open_time'].head(3).tolist()}"

    def test_aligned_on_second(self, klines):
        """
        Chaque bougie commence pile sur une seconde (pas de millisecondes résiduelles).

        Pourquoi : la grille temporelle doit être régulière pour que « une ligne = une
        seconde » soit vrai après le remplissage des trous. Un open_time à 12:00:00.500
        ne correspondrait à aucune case de la grille et serait perdu ou dupliqué.
        """
        _, df = klines
        misaligned = df["open_time"] != df["open_time"].dt.floor("s")
        assert not misaligned.any(), f"{misaligned.sum()} open_time non alignés sur la seconde"

    def test_close_time_consistent(self, klines):
        """
        Chaque bougie dure une seconde : close_time = open_time + 999 ms.

        Pourquoi : la durée d'une bougie conditionne la comparabilité de son volume et de
        son amplitude (high - low) avec celles des autres. Une bougie de 3 secondes aurait
        un volume anormalement grand et fausserait les features de volume et de volatilité.

        Exception légitime, la bougie tronquée par un arrêt du marché : quand Binance
        interrompt le trading (maintenance, panne) au milieu d'une seconde, la dernière
        bougie est fermée à l'instant de l'arrêt et dure moins de 999 ms (parfois 0 ms).
        Ses données sont réelles. On l'accepte si et seulement si elle est suivie d'un
        trou, c'est-à-dire si la seconde suivante est absente des données. Le trou peut
        être long (maintenance) ou court (interruption de quelques secondes).

        Ces bougies tronquées marquent une fin de série : lors de la construction des
        features, aucune fenêtre ne doit traverser le trou qui les suit.

        Le test échoue pour toute autre anomalie :
        - bougie de plus d'une seconde : agrégation incorrecte ;
        - bougie tronquée suivie d'une bougie normale : aucun arrêt ne l'explique, c'est
          une donnée suspecte.
        """
        path, df = klines
        delta = df["close_time"] - df["open_time"]
        too_long = delta >= pd.Timedelta(seconds=1)
        truncated = delta < pd.Timedelta(milliseconds=999)

        present = pd.DatetimeIndex(df["open_time"])
        next_missing = ~(df["open_time"] + pd.Timedelta(seconds=1)).isin(present)
        unexplained = truncated & ~next_missing

        def show(mask):
            return df.loc[mask, ["open_time", "close_time"]].head(5).to_string(index=False)

        errors = []
        if too_long.any():
            errors.append(f"{too_long.sum()} bougie(s) de plus d'une seconde :\n{show(too_long)}")
        if unexplained.any():
            errors.append(f"{unexplained.sum()} bougie(s) tronquée(s) sans arrêt du marché "
                          f"juste après :\n{show(unexplained)}")
        message = f"{path.name}\n" + "\n".join(errors)
        assert not errors, message


# =========================================================================== klines 1s : trous

@pytest.mark.klines
@pytest.mark.completude
class TestKlinesGaps:
    """
    Les secondes manquantes sont-elles explicables ?

    Deux natures de trous, voir gaps.py pour le détail :
    - courts (<= SHORT_GAP_MAX_SECONDS) : secondes sans aucun trade, on les comble ;
    - longs : exchange fermé (maintenance, panne), on coupe la série à cet endroit.
    """

    def test_short_gaps_are_rare(self, klines, kline_gaps):
        """
        Les trous courts représentent une faible part du mois (MAX_SHORT_GAP_RATIO).

        Pourquoi : quelques secondes sans trade sont normales, même sur BTCUSDT. Mais si
        elles deviennent nombreuses, c'est que le marché était peu liquide ou que le
        fichier est incomplet. Dans les deux cas, les secondes comblées artificiellement
        (prix reporté, volume nul) deviennent une part significative des données : le
        modèle apprendrait surtout du « rien ne se passe » fabriqué par nous.
        """
        path, _ = klines
        short, _, n_expected = kline_gaps
        missing = int(short["duration_s"].sum())
        ratio = missing / n_expected
        assert ratio <= MAX_SHORT_GAP_RATIO, (
            f"{path.name} : {missing} s manquantes en trous courts ({ratio:.4%}, max {MAX_SHORT_GAP_RATIO:.4%})\n"
            f"{len(short)} trous courts, les plus longs :\n{describe_gaps(short)}"
        )

    def test_long_gaps_are_known(self, klines, kline_gaps, known_gaps):
        """
        Chaque trou long figure dans le registre data/known_gaps.csv.

        Pourquoi : un trou long a deux explications possibles, et elles appellent des
        réactions opposées :
        - l'exchange était fermé : le trou est réel, on l'accepte et on coupe la série ;
        - le téléchargement a raté une partie du fichier : il faut re-télécharger.
        Le registre sert à trancher. Un trou long qui n'y figure pas est une nouveauté à
        examiner. Une fois vérifié (par exemple en cherchant une annonce de maintenance
        Binance à cette date), on l'ajoute au registre avec `python gaps.py` et on peut
        renseigner la colonne `reason`.

        Le registre est ensuite réutilisé pour construire les jeux de données : aucune
        fenêtre de features ni variable cible ne doit chevaucher un trou long.
        """
        path, _ = klines
        _, long_, _ = kline_gaps
        known = {(r.start, r.end) for r in known_gaps.itertuples() if r.dataset == dataset_of(path)}
        unknown = long_[[(r.start, r.end) not in known for r in long_.itertuples()]]
        assert unknown.empty, (
            f"{path.name} : {len(unknown)} trou(s) long(s) absent(s) de {known_gaps_path()}\n"
            f"{describe_gaps(unknown)}\n"
            "Vérifie qu'il s'agit bien d'une fermeture de Binance, puis lance `python gaps.py`. "
            "Sinon, supprime le fichier du mois et re-télécharge-le."
        )


# =========================================================================== klines 1s : valeurs

@pytest.mark.klines
@pytest.mark.coherence
class TestKlinesPrices:
    """Les prix sont-ils économiquement plausibles ?"""

    def test_prices_positive(self, klines):
        """
        Tous les prix sont strictement positifs.

        Pourquoi : un prix nul ou négatif est impossible sur le spot et casse les calculs
        de rendements logarithmiques (log(0) = -inf), très utilisés comme features.
        """
        _, df = klines
        prices = df[["open", "high", "low", "close"]]
        assert (prices > 0).all().all(), "prix nuls ou négatifs"

    def test_ohlc_consistency(self, klines):
        """
        Le plus haut est au-dessus de l'ouverture et de la clôture, le plus bas en dessous.

        Pourquoi : c'est la définition même d'une bougie. Si ce n'est pas vrai, les
        colonnes ont été mélangées (erreur de parsing) ou la donnée est corrompue. Les
        features de volatilité (high - low) et les simulations de stop-loss, qui
        regardent si le prix a touché un niveau pendant la seconde, seraient fausses.
        """
        _, df = klines
        bad_high = df["high"] < df[["open", "close"]].max(axis=1)
        bad_low = df["low"] > df[["open", "close"]].min(axis=1)
        assert not bad_high.any(), f"{bad_high.sum()} bougies avec high < max(open, close)"
        assert not bad_low.any(), f"{bad_low.sum()} bougies avec low > min(open, close)"

    def test_no_absurd_price_jump(self, klines):
        """
        Pas de variation de prix supérieure à MAX_PRICE_JUMP (10 %) d'une seconde à l'autre.

        Pourquoi : même lors des krachs les plus violents, le bitcoin ne perd pas 10 % en
        une seconde sur Binance spot. Un tel saut indique presque toujours une erreur
        (prix d'une autre paire, virgule décalée). Un seul point aberrant suffit à fausser
        la normalisation des features et peut devenir le « trade parfait » que le modèle
        cherchera ensuite à reproduire. Avec un levier x5, un tel saut en réel signifierait
        une liquidation : il faut en être certain avant de l'accepter.
        """
        _, df = klines
        ret = np.log(df["close"]).diff().abs()
        jumps = df.loc[ret > np.log1p(MAX_PRICE_JUMP), "open_time"]
        assert jumps.empty, f"{len(jumps)} sauts de prix > {MAX_PRICE_JUMP:.0%} en 1 s, ex. {jumps.head(3).tolist()}"



@pytest.mark.klines
@pytest.mark.volumes
class TestKlinesVolumes:
    """Les volumes sont-ils cohérents entre eux et avec les prix ?"""

    def test_volumes(self, klines):
        """
        Les volumes sont positifs et le volume acheteur ne dépasse pas le volume total.

        Pourquoi : le déséquilibre acheteurs / vendeurs (taker_buy_base / volume) est une
        des features les plus informatives à court terme. Il doit rester entre 0 et 1. Un
        volume acheteur supérieur au volume total signalerait des colonnes inversées.
        """
        _, df = klines
        for col in ("volume", "quote_volume", "taker_buy_base", "taker_buy_quote", "n_trades"):
            assert (df[col] >= 0).all(), f"{col} contient des valeurs négatives"
        assert (df["taker_buy_base"] <= df["volume"] * (1 + 1e-9)).all(), "taker_buy_base > volume"
        assert (df["taker_buy_quote"] <= df["quote_volume"] * (1 + 1e-9)).all(), "taker_buy_quote > quote_volume"

    def test_trades_imply_volume(self, klines):
        """
        Une bougie avec des trades a forcément un volume non nul.

        Pourquoi : c'est une cohérence interne simple. Si elle est violée, les colonnes
        n_trades et volume ne décrivent pas la même chose (décalage de colonnes).
        """
        _, df = klines
        bad = (df["n_trades"] > 0) & (df["volume"] <= 0)
        assert not bad.any(), f"{bad.sum()} bougies avec des trades mais un volume nul"

    def test_quote_volume_matches_price(self, klines):
        """
        Le prix moyen implicite (quote_volume / volume) est compris entre low et high.

        Pourquoi : quote_volume est le montant en USDT échangé, volume la quantité de BTC.
        Leur rapport est donc le prix moyen des trades de la seconde (VWAP), qui ne peut
        pas sortir de la fourchette [low, high]. C'est un contrôle croisé fort entre les
        colonnes de prix et de volume : s'il échoue, l'une des deux familles est fausse.
        """
        _, df = klines
        v = df[df["volume"] > 0]
        vwap = v["quote_volume"] / v["volume"]
        bad = (vwap < v["low"] * 0.999) | (vwap > v["high"] * 1.001)
        assert not bad.any(), f"{bad.sum()} bougies avec un prix moyen hors de [low, high]"

# =========================================================================== couverture globale

class TestCoverage:
    """L'ensemble des fichiers forme-t-il un historique continu et à jour ?"""

    @pytest.mark.parametrize("dataset", list(DATASETS))
    @pytest.mark.completude
    @pytest.mark.klines
    @pytest.mark.funding
    def test_no_missing_month(self, dataset):
        """
        Aucun mois manquant entre le premier et le dernier fichier téléchargé.

        Pourquoi : download_binance.py ignore silencieusement un mois indisponible (erreur
        404). Un mois manquant au milieu de l'historique passerait inaperçu, mais créerait
        un trou d'un mois que les tests par fichier ne peuvent pas voir, puisqu'il n'y a
        pas de fichier à tester.
        """
        files = files_of(dataset, DATA)
        if not files:
            pytest.skip(f"aucun fichier pour {dataset}")
        present = {month_of(p) for p in files}
        m, last = min(present), max(present)
        missing = []
        while m <= last:
            if m not in present:
                missing.append(f"{m:%Y-%m}")
            m = next_month(m)
        assert not missing, f"{dataset} : mois manquants {missing}"

    @pytest.mark.parametrize("dataset", SECOND_DATASETS)
    @pytest.mark.completude
    @pytest.mark.klines
    def test_klines_up_to_date(self, dataset):
        """
        Les données 1s vont au moins jusqu'à avant-hier.

        Pourquoi : Binance publie les fichiers journaliers avec un jour de décalage. Au-delà,
        c'est que le téléchargement n'a pas été relancé. Entraîner ou évaluer sur des
        données périmées fait passer à côté du régime de marché actuel, qui est celui
        dans lequel le bot tradera.
        """
        files = files_of(dataset, DATA)
        if not files:
            pytest.skip(f"aucun fichier pour {dataset}")
        last = pd.read_parquet(files[-1], columns=["open_time"])["open_time"].max()
        limit = utc(date.today() - timedelta(days=2))
        assert last >= limit, f"{dataset} : dernière donnée au {last}, relance download_binance.py"

    @pytest.mark.structure
    @pytest.mark.klines
    @pytest.mark.funding
    def test_no_leftover_tmp_files(self):
        """
        Aucun fichier .tmp ne traîne dans le dossier de données.

        Pourquoi : download_binance.py écrit d'abord dans un .tmp puis le renomme, pour
        qu'une interruption ne laisse jamais un Parquet à moitié écrit. Un .tmp restant
        signifie donc qu'une écriture a été interrompue : le mois correspondant est
        peut-être incomplet et doit être re-téléchargé.
        """
        tmp = list(DATA.rglob("*.tmp"))
        assert not tmp, f"fichiers temporaires restants : {tmp}"


@pytest.mark.registre
@pytest.mark.completude
class TestKnownGapsRegistry:
    """Le registre des trous longs est-il lui-même cohérent ?"""

    def test_registry_well_formed(self, known_gaps):
        """
        Chaque entrée du registre a une fin après son début et une durée cohérente.

        Pourquoi : le registre est édité à la main (colonne reason) et sera utilisé pour
        couper les séries lors de la construction des datasets. Une ligne corrompue
        ferait couper au mauvais endroit, ou pas du tout.
        """
        if known_gaps.empty:
            pytest.skip("registre vide ou absent, lance `python gaps.py`")
        assert (known_gaps["end"] >= known_gaps["start"]).all(), "trou avec end < start"
        computed = ((known_gaps["end"] - known_gaps["start"]) // pd.Timedelta(seconds=1) + 1).astype("int64")
        assert (computed == known_gaps["duration_s"]).all(), "duration_s incohérent avec start / end"

    def test_registry_no_overlap(self, known_gaps):
        """
        Les trous du registre ne se chevauchent pas.

        Pourquoi : deux trous qui se chevauchent sont le signe d'une édition manuelle
        erronée ou d'un registre construit avec deux seuils différents. Cela fausserait
        le décompte des heures de fermeture et le découpage des séries.
        """
        if known_gaps.empty:
            pytest.skip("registre vide ou absent, lance `python gaps.py`")
        for dataset, g in known_gaps.groupby("dataset"):
            g = g.sort_values("start")
            overlap = g["start"].iloc[1:].values <= g["end"].iloc[:-1].values
            assert not overlap.any(), f"{dataset} : {overlap.sum()} trous qui se chevauchent"


# =========================================================================== vérification croisée

@pytest.fixture(scope="module", params=months_in_both("futures_klines_1s", "futures_klines_1m"))
def perp_1s_vs_1m(request):
    """
    Pour un mois donné : les bougies 1 s reconstruites agrégées en 1 min, et les bougies
    1 min officielles de Binance (limitées aux minutes où il y a eu des trades).
    """
    path_1s, path_1m = request.param
    s = pd.read_parquet(path_1s)
    ours = s.groupby(s["open_time"].dt.floor("min")).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), quote_volume=("quote_volume", "sum"),
        n_trades=("n_trades", "sum"), taker_buy_base=("taker_buy_base", "sum"),
    )
    official = pd.read_parquet(path_1m)
    official = official[official["volume"] > 0].set_index("open_time")[ours.columns]
    return path_1s.name, ours, official


@pytest.mark.perp
@pytest.mark.verification
class TestPerpetualVsOfficial1m:
    """
    Les bougies 1 s du perpétuel, que nous reconstruisons nous-mêmes à partir des
    aggTrades, sont-elles justes ?

    Binance ne publie pas de bougies 1 s pour les futures, mais publie des bougies 1 min,
    calculées de son côté. En agrégeant nos bougies 1 s par minute, on doit retrouver
    exactement les bougies officielles. C'est un contrôle par une source indépendante :
    contrairement aux autres tests, qui vérifient la cohérence interne des données, il
    détecte aussi une erreur plausible mais fausse (trade oublié, mauvais sens
    acheteur / vendeur, erreur d'arrondi, décalage d'horodatage).
    """

    def test_same_minutes(self, perp_1s_vs_1m):
        """
        Les minutes avec des trades sont les mêmes des deux côtés.

        Pourquoi : une minute présente chez Binance mais absente chez nous signifie que des
        trades ont été perdus (morceau de fichier mal lu, jour manquant). L'inverse
        signifierait des trades horodatés dans la mauvaise minute, voire inventés.
        """
        name, ours, official = perp_1s_vs_1m
        missing = official.index.difference(ours.index)
        extra = ours.index.difference(official.index)
        assert missing.empty and extra.empty, (
            f"{name} : {len(missing)} minute(s) absente(s) de nos données "
            f"(ex. {list(missing[:3])}), {len(extra)} minute(s) en trop (ex. {list(extra[:3])})"
        )

    def test_prices_match(self, perp_1s_vs_1m):
        """
        open, high, low et close coïncident à la minute près.

        Pourquoi : les prix d'ouverture et de clôture dépendent de l'ordre des trades, le
        plus haut et le plus bas de leur exhaustivité. Une différence révèle un tri
        incorrect ou des trades manquants, ce qui fausserait directement les rendements
        et les simulations de stop-loss du backtest.
        """
        name, ours, official = perp_1s_vs_1m
        common = ours.index.intersection(official.index)
        errors = []
        for col in ("open", "high", "low", "close"):
            a, b = ours.loc[common, col], official.loc[common, col]
            bad = ~np.isclose(a, b, rtol=1e-9, atol=0)
            if bad.any():
                errors.append(f"{col} : {bad.sum()} minutes différentes, ex. {list(common[bad][:3])}")
        assert not errors, f"{name}\n" + "\n".join(errors)

    def test_volumes_match(self, perp_1s_vs_1m):
        """
        Volumes, nombre de trades et volume acheteur coïncident.

        Pourquoi : le volume acheteur (taker_buy_base) dépend de l'interprétation du
        champ is_buyer_maker des aggTrades. Une inversion de ce champ passerait tous les
        tests de cohérence interne, mais inverserait le sens de la pression
        acheteurs / vendeurs, une des features les plus importantes du modèle.
        """
        name, ours, official = perp_1s_vs_1m
        common = ours.index.intersection(official.index)
        errors = []
        for col in ("volume", "quote_volume", "n_trades", "taker_buy_base"):
            a, b = ours.loc[common, col], official.loc[common, col]
            bad = ~np.isclose(a, b, rtol=1e-6, atol=1e-8)
            if bad.any():
                errors.append(f"{col} : {bad.sum()} minutes différentes, ex. {list(common[bad][:3])}")
        assert not errors, f"{name}\n" + "\n".join(errors)


# =========================================================================== funding

@pytest.mark.funding
class TestFunding:
    """Le funding rate est un coût direct des positions à levier : il doit être fiable."""

    @pytest.mark.structure
    def test_not_empty(self, funding):
        """
        Le fichier contient au moins un paiement.

        Pourquoi : avec 3 paiements par jour, un mois en compte environ 90. Un fichier
        vide indique un téléchargement raté, et le backtest sous-estimerait le coût de
        détention des positions.
        """
        path, df = funding
        assert len(df) > 0, f"{path.name} est vide"

    @pytest.mark.structure
    def test_no_nan(self, funding):
        """
        Aucune valeur manquante.

        Pourquoi : un taux manquant serait probablement traité comme 0 par le backtest,
        c'est-à-dire un paiement gratuit qui n'a pas existé.
        """
        _, df = funding
        assert not df.isna().any().any()

    @pytest.mark.chronologie
    def test_no_duplicates_and_sorted(self, funding):
        """
        Chaque paiement n'apparaît qu'une fois, dans l'ordre chronologique.

        Pourquoi : un paiement en double serait facturé deux fois dans le backtest, et
        des paiements mal ordonnés seraient attribués aux mauvaises positions.
        """
        _, df = funding
        assert not df["calc_time"].duplicated().any()
        assert df["calc_time"].is_monotonic_increasing

    @pytest.mark.chronologie
    def test_timestamps_in_file_month(self, funding):
        """
        Tous les paiements appartiennent au mois du nom de fichier.

        Pourquoi : même raison que pour les klines, et détection des erreurs d'unité
        de timestamp.
        """
        path, df = funding
        start, end = utc(month_of(path)), utc(next_month(month_of(path)))
        assert df["calc_time"].between(start, end, inclusive="left").all()

    @pytest.mark.completude
    def test_regular_interval(self, funding):
        """
        L'écart entre deux paiements ne dépasse pas l'intervalle déclaré (8 h pour BTCUSDT).

        Pourquoi : un écart plus grand signifie un paiement manquant. Le backtest ferait
        alors comme si une position tenue à ce moment-là n'avait rien payé ni reçu.
        """
        _, df = funding
        gaps = df["calc_time"].dt.round("min").diff().dropna()
        allowed = pd.to_timedelta(df["funding_interval_hours"].iloc[1:].values, unit="h")
        bad = gaps.values > allowed
        assert not bad.any(), f"{bad.sum()} écarts anormaux entre paiements de funding"

    @pytest.mark.coherence
    def test_rate_bounds(self, funding):
        """
        Le taux reste dans des bornes réalistes (moins de 3 % par paiement en valeur absolue).

        Pourquoi : un taux typique est de l'ordre de 0,01 %. Binance plafonne le funding,
        et même dans les phases les plus extrêmes il reste bien en dessous de 3 %. Une
        valeur plus grande est presque sûrement une erreur d'unité (pourcentage lu comme
        fraction), qui ferait exploser le coût simulé des positions.
        """
        _, df = funding
        r = df["last_funding_rate"].abs()
        assert (r < 0.03).all(), f"funding rate aberrant : max {r.max():.4%}"