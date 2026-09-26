# Bot de trading BTC — données et qualité

Projet de bot de trading sur le **perpétuel BTCUSDT de Binance**, avec un levier maximum de ×5. L'objectif final est d'entraîner trois modèles :

1. **quand entrer** sur le marché ;
2. **quelle position prendre** (long ou short) ;
3. **quand sortir**.

Les expériences seront suivies avec un serveur **MLflow** local.

Ce dépôt contient pour l'instant la **première brique** : récupérer gratuitement un historique à la seconde depuis 2020, puis vérifier sa qualité.

> Vocabulaire financier (kline, perpétuel, funding, taker…) : voir [GLOSSAIRE.md](GLOSSAIRE.md).

---

## Démarrage rapide

```bash
# 1. Environnement
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt

# 2. Données (plusieurs heures la première fois, plusieurs dizaines de Go)
python download_binance.py

# 3. Registre des trous longs (à relancer après chaque téléchargement)
python gaps.py

# 4. Tests de qualité
pytest
```

Toutes les commandes se lancent **depuis la racine du dépôt**.

---

## Structure

```
.
├── download_binance.py      # téléchargement incrémental depuis Binance Vision
├── gaps.py                  # détection des secondes manquantes + registre des trous longs
├── tests/
│   └── test_data_quality.py # tests de qualité des données (pytest)
├── pytest.ini               # configuration pytest et déclaration des catégories de tests
├── requirements.txt
├── GLOSSAIRE.md             # définitions des termes financiers
└── data/                    # non versionné, sauf known_gaps.csv
    ├── raw/<dataset>/<dataset>_AAAA-MM.parquet
    ├── known_gaps.csv       # registre des trous longs (versionné)
    └── _downloads/          # ZIP temporaires, supprimés après traitement
```

---

## Les données

Source : [Binance Vision](https://data.binance.vision), le dépôt public et gratuit de Binance. Aucune clé API n'est nécessaire.

| Dataset | Marché | Contenu | Fréquence | Rôle |
|---|---|---|---|---|
| `futures_klines_1s` | perpétuel USD-M | bougies OHLCV + volume acheteur | 1 s | **données de travail principales** |
| `futures_klines_1m` | perpétuel USD-M | bougies officielles Binance | 1 min | vérification de `futures_klines_1s` |
| `futures_funding` | perpétuel USD-M | funding rate | 8 h | coût de détention des positions, feature |
| `spot_klines_1s` | spot | bougies OHLCV + volume acheteur | 1 s | feature complémentaire (écart spot / perpétuel) |

Toutes les bougies 1 s ont les mêmes colonnes : `open_time`, `open`, `high`, `low`, `close`, `volume`, `close_time`, `quote_volume`, `n_trades`, `taker_buy_base`, `taker_buy_quote`. Toutes les dates sont en **UTC**.

### Comment fonctionne le téléchargement

- **Un fichier Parquet par mois**, depuis janvier 2020. Le mois en cours est complété jour par jour, jusqu'à la veille.
- **Incrémental** : un mois déjà présent sur disque n'est pas re-téléchargé, et une ligne déjà présente n'est jamais ajoutée deux fois. On peut interrompre le script et le relancer à tout moment.
- **Intégrité** : le SHA256 de chaque ZIP est vérifié, et chaque fichier est écrit de façon atomique, donc jamais laissé à moitié écrit.
- **Bougies 1 s du perpétuel reconstruites** : Binance ne publie pas de bougies 1 s pour les futures. Le script les calcule à partir des *aggTrades* (toutes les transactions), en lisant les fichiers par morceaux pour limiter la mémoire à environ 1 à 2 Go.

```bash
python download_binance.py                                        # tous les datasets
python download_binance.py --datasets futures_klines_1s futures_klines_1m
python download_binance.py -v                                     # affiche aussi les mois ignorés
python download_binance.py --datasets spot_klines_1s --force      # re-télécharge tout (long)
```

Pour re-télécharger **un seul** mois, supprime son fichier Parquet puis relance le script.

---

## Les trous dans les données

Binance ne publie pas de bougie pour une seconde sans aucun trade. Les secondes manquantes sont de deux natures, qu'il faut traiter différemment :

| | Durée | Cause | Traitement dans les datasets |
|---|---|---|---|
| **Trou court** | ≤ 60 s | aucun trade pendant quelques secondes | reporter le dernier prix, volume à 0 |
| **Trou long** | > 60 s | exchange fermé (maintenance, panne) | **couper la série** : aucune fenêtre de features ni variable cible ne doit traverser le trou |

`python gaps.py` recense tous les trous longs dans `data/known_gaps.csv`. La colonne `reason` peut être remplie à la main (par exemple « maintenance Binance ») ; elle est conservée quand on régénère le fichier. Un trou long **absent** du registre fait échouer les tests : c'est peut-être un téléchargement raté plutôt qu'une vraie fermeture.

**Bougies tronquées** : quand Binance arrête le trading au milieu d'une seconde, la dernière bougie dure moins de 999 ms. Elle est valide, et les tests l'acceptent si la seconde suivante est manquante.

---

## Les tests de qualité

```bash
pytest                                  # tout
pytest -m perp                          # seulement le perpétuel
pytest -m "completude or chronologie"   # une ou plusieurs catégories
pytest -m verification -k "2024-03"     # une catégorie sur un seul mois
pytest --lf                             # seulement ce qui a échoué la dernière fois
```

| Catégorie | Ce qu'elle vérifie |
|---|---|
| `structure` | fichiers non vides, colonnes, types, absence de NaN, pas de `.tmp` restant |
| `chronologie` | pas de doublons, ordre chronologique, bon mois, alignement sur la seconde, durée des bougies |
| `completude` | trous courts rares, trous longs déclarés, aucun mois manquant, données à jour, funding régulier |
| `coherence` | prix positifs, high/low cohérents avec open/close, pas de saut de prix absurde, funding dans des bornes réalistes |
| `volumes` | volumes positifs, volume acheteur ≤ volume total, prix moyen (VWAP) entre low et high |
| `registre` | `known_gaps.csv` bien formé, sans chevauchement |
| `verification` | bougies 1 s du perpétuel agrégées en 1 min = bougies 1 min officielles de Binance |

| Étiquette de dataset | Portée |
|---|---|
| `klines` | bougies 1 s (spot et perpétuel) |
| `spot` / `perp` | bougies 1 s d'un seul marché |
| `funding` | funding rate |

Chaque test explique dans sa docstring **pourquoi** il existe, c'est-à-dire quelle erreur de modèle il évite. Le tableau « quand relancer quelle catégorie » se trouve en tête de `tests/test_data_quality.py`.

Les seuils se règlent par variables d'environnement, par exemple `MAX_SHORT_GAP_RATIO=0.01 pytest -m completude`. La liste complète est en tête du fichier de tests.

---

## Points d'attention pour la suite

- **Entraîner sur le perpétuel**, le marché où le bot passera ses ordres. Le prix du spot en est proche, mais pas identique.
- **Ne pas utiliser les prix bruts** comme features : le BTC est passé d'environ 5 000 $ à bien plus. Utiliser des rendements et des variations relatives.
- **Normaliser les volumes** : ils varient beaucoup selon les périodes. Par exemple, Binance a supprimé les frais sur les paires BTC du spot de mi-2022 à mars 2023.
- **Intégrer tous les coûts** dans les variables cibles et le backtest : frais (environ 0,05 % par ordre taker), funding (payé toutes les 8 h si la position est ouverte) et spread (inconnu, car les données ne contiennent que des trades, pas le carnet d'ordres).
- **Découper train / test par période**, jamais au hasard, pour éviter que le modèle voie le futur.

---

## Feuille de route

- [x] Téléchargement incrémental des données (spot, perpétuel, funding)
- [x] Tests de qualité et registre des trous
- [ ] Construction des features
- [ ] Construction des trois variables cibles (entrée, direction, sortie)
- [ ] Serveur MLflow local et entraînement des modèles
- [ ] Backtest avec frais, funding et levier