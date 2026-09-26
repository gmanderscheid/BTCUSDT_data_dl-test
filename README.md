# Bot de trading BTC — données et qualité

Projet de bot de trading sur le **perpétuel BTCUSDT de Binance**, avec un levier maximum de ×5. L'objectif final est d'entraîner trois modèles :

1. **quand entrer** sur le marché ;
2. **quelle position prendre** (long ou short) ;
3. **quand sortir**.

Les expériences seront suivies avec un serveur **MLflow** local.

Ce dépôt contient pour l'instant la **première brique** : récupérer gratuitement un historique à la seconde depuis 2020, puis vérifier sa qualité.

> Vocabulaire financier (kline, perpétuel, funding, taker…) : voir [GLOSSAIRE.md](GLOSSAIRE.md). Historique des choix et de la revue : voir [SESSION_QUALITE_DONNEES.md](SESSION_QUALITE_DONNEES.md).

---

## Démarrage rapide

```bash
# 1. Environnement
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
pytest -m unit                     # vérifie le code du pipeline (quelques secondes, sans données)

# 2. Données (plusieurs heures la première fois, plusieurs dizaines de Go)
python download_binance.py

# 3. Détection des anomalies (à relancer après chaque téléchargement)
python gaps.py                     # trous longs -> data/known_gaps.csv (statut candidate)
python diagnose_perp.py --register # minutes irréparables -> data/known_minute_mismatches.csv

# 4. Examen humain des anomalies, puis approbation
python approve.py --list
python approve.py gaps 2021-04-25 "maintenance annoncée par Binance"

# 5. Validation complète
pytest
```

Toutes les commandes se lancent **depuis la racine du dépôt**.

---

## Structure

```
.
├── download_binance.py      # téléchargement, reconstruction 1 s, dédoublonnage, réparations, provenance
├── gaps.py                  # détection des trous (fusionnés entre deux mois) + registre des trous longs
├── diagnose_perp.py         # diagnostic mois par mois : perpétuel 1 s vs bougies 1 min officielles
├── approve.py               # approbation des anomalies examinées (registres)
├── quality_checks.py        # contrôles de qualité et seuils, partagés par pytest et le diagnostic
├── data_contract.py         # contrat de couverture : datasets, périodes et référence attendus
├── tests/
│   ├── test_data_quality.py # tests des DONNÉES réelles
│   └── test_pipeline.py     # tests UNITAIRES du code, sur données synthétiques
├── pytest.ini               # configuration pytest et déclaration des catégories de tests
├── requirements.txt
├── GLOSSAIRE.md             # définitions des termes financiers
└── data/                    # non versionné, sauf les fichiers de suivi ci-dessous
    ├── raw/<dataset>/<dataset>_AAAA-MM.parquet
    ├── known_gaps.csv               # registre des trous longs (versionné)
    ├── known_minute_mismatches.csv  # registre des minutes irréparables du perpétuel (versionné)
    ├── manifest.csv                 # provenance de chaque fichier intégré (versionné)
    ├── repair_attempts.csv          # réparations déjà tentées (versionné)
    └── _downloads/                  # ZIP temporaires, supprimés après traitement
```

---

## Les données

Source : [Binance Vision](https://data.binance.vision), le dépôt public et gratuit de Binance. Aucune clé API n'est nécessaire.

| Dataset | Marché | Contenu | Fréquence | Rôle |
|---|---|---|---|---|
| `futures_klines_1s` | perpétuel USD-M | bougies OHLCV + volume acheteur | 1 s | **données de travail principales** |
| `futures_klines_1m` | perpétuel USD-M | bougies officielles Binance | 1 min | référence pour vérifier `futures_klines_1s` |
| `futures_funding` | perpétuel USD-M | funding rate | 8 h | coût de détention des positions, feature |
| `spot_klines_1s` | spot | bougies OHLCV + volume acheteur | 1 s | feature complémentaire (écart spot / perpétuel) |

Toutes les bougies 1 s ont les mêmes colonnes : `open_time`, `open`, `high`, `low`, `close`, `volume`, `close_time`, `quote_volume`, `n_trades`, `taker_buy_base`, `taker_buy_quote`. Toutes les dates sont en **UTC**.

**`n_trades` du perpétuel est une approximation par excès** : il est calculé à partir des plages d'identifiants des aggTrades et dépasse la valeur officielle même quand les volumes concordent. Ne pas l'utiliser comme un nombre exact.

### Comment fonctionne le téléchargement

- **Un fichier Parquet par mois**, depuis janvier 2020. Le mois en cours est complété jour par jour, jusqu'à la veille.
- **Incrémental** : un mois déjà présent n'est pas re-téléchargé, et une ligne déjà présente n'est jamais ajoutée deux fois. On peut interrompre le script et le relancer à tout moment.
- **`--force` remplace** : le mois re-téléchargé remplace entièrement l'ancien contenu (utile après une correction du code de reconstruction).
- **Checksum obligatoire** : le SHA256 de chaque ZIP est vérifié. Un checksum faux, vide, mal formé ou indisponible fait échouer le téléchargement (`--allow-missing-checksum` accepte explicitement l'absence de `.CHECKSUM`, avec avertissement).
- **Bougies 1 s du perpétuel reconstruites** à partir des *aggTrades*, lus par morceaux (environ 1 à 2 Go de RAM). Les aggTrades en double sont retirés. L'ordre des identifiants, dont dépendent open et close, est **vérifié** : un identifiant inédit hors ordre arrête la reconstruction au lieu d'être supprimé silencieusement.
- **Réparations** : les journées absentes ou incomplètes des fichiers mensuels sont récupérées dans les fichiers journaliers. Pour le perpétuel, une journée est aussi re-téléchargée si son volume diffère de plus de 0,5 % de la référence officielle. Une journée n'est remplacée que si le fichier journalier couvre au moins les mêmes instants. Chaque tentative est notée dans `data/repair_attempts.csv` et n'est pas refaite (sauf `--retry-repairs`).
- **Provenance** : chaque fichier intégré est tracé dans `data/manifest.csv` (source, SHA256, date, mode fusion / remplacement, empreinte du code de transformation).

```bash
python download_binance.py                                        # tous les datasets
python download_binance.py --datasets futures_klines_1s futures_klines_1m
python download_binance.py -v                                     # affiche aussi les mois ignorés
python download_binance.py --datasets spot_klines_1s --force      # re-télécharge et remplace tout (long)
```

Pour re-télécharger **un seul** mois, supprime son fichier Parquet puis relance le script. Pour figer les versions de bibliothèques utilisées lors d'une validation : `pip freeze > requirements.lock`.

---

## Les trous et les exceptions

Binance ne publie pas de bougie pour une seconde sans aucun trade. Deux traitements, selon la durée :

| | Durée | Traitement dans les datasets |
|---|---|---|
| **Trou court** | ≤ 60 s | reporter le dernier prix, volume à 0 |
| **Trou long** | > 60 s | **couper la série** : aucune fenêtre de features ni variable cible ne doit traverser le trou |

Le seuil de 60 s est une **règle de traitement** : il ne prouve ni qu'un trou long est une maintenance, ni qu'un trou court correspond à zéro transaction. Les trous sont fusionnés aux changements de mois avant d'être classés.

**Détection ≠ approbation.** Les anomalies détectées sont inscrites dans deux registres avec le statut `candidate` :
- `data/known_gaps.csv` (trous longs), par `python gaps.py` ;
- `data/known_minute_mismatches.csv` (minutes du perpétuel absentes d'un côté, ou zones jugées non fiables), par `python diagnose_perp.py --register`.

Les tests **échouent tant qu'une anomalie n'est pas examinée**. Après vérification (maintenance annoncée ? téléchargement raté ? volume en jeu ?), on l'approuve avec une raison :

```bash
python approve.py --list
python approve.py gaps 2021-04-25 "maintenance annoncée par Binance"
python approve.py minutes 2024-10-28 "bougies officielles absentes, fichier journalier inclus"
```

Une journée entière manquante n'est jamais une maintenance : c'est un fichier incomplet. Les tests échouent aussi sur une exception devenue obsolète, et limitent le volume qu'un registre peut exclure. **Les registres serviront de masque lors de la construction des features.**

**Bougies tronquées** : quand Binance arrête le trading au milieu d'une seconde, la dernière bougie dure moins de 999 ms. Elle est acceptée si la seconde suivante est absente, y compris dans le fichier du mois suivant. Une durée négative n'est jamais acceptée.

---

## Les tests

Deux familles :
- **`tests/test_pipeline.py`** (étiquette `unit`) : tests unitaires du code, sur données synthétiques à résultat connu. Ils vérifient la reconstruction (valeurs calculées à la main, tailles de morceaux, doublons, ordre), le téléchargement (checksum), la fusion, le remplacement, la réparation, et que chaque contrôle de qualité **détecte bien** les défauts qu'il vise (cas négatifs) sans rejeter les écarts légitimes (cas positifs).
- **`tests/test_data_quality.py`** : contrôles appliqués aux vraies données.

```bash
pytest                                  # tout
pytest -m unit                          # tests unitaires seulement
pytest -m perp                          # seulement le perpétuel
pytest -m "completude or chronologie"   # une ou plusieurs catégories
pytest -m verification -k "2024-03"     # une catégorie sur un seul mois
pytest --lf                             # seulement ce qui a échoué la dernière fois
```

| Catégorie | Ce qu'elle vérifie |
|---|---|
| `structure` | fichiers non vides, colonnes, types, valeurs finies, `n_trades` entier, pas de `.tmp` |
| `chronologie` | doublons, ordre, bon mois, alignement, durée des bougies (y compris entre deux mois) |
| `completude` | **contrat de couverture** (datasets, premier et dernier mois, mois manquants, fraîcheur, référence 1 min), trous courts / longs, échéancier complet du funding |
| `coherence` | prix positifs, OHLC, sauts de prix (rendement simple, symétrique), ordre de grandeur du funding |
| `volumes` | volumes positifs, acheteur ≤ total, volumes nuls cohérents, VWAP et VWAP acheteur |
| `registre` | structure des deux registres, anomalies examinées, pas d'exception obsolète |
| `verification` | perpétuel 1 s agrégé en 1 min contre les bougies officielles : minutes identiques, extrêmes cohérents (ni inventés ni disparus) avec les minutes voisines exactes, close, dérive du volume cumulé, volumes par jour et par heure, volume du mois, part des minutes réellement comparées |
| `unit` | tests unitaires du pipeline et des contrôles |

| Étiquette de dataset | Portée |
|---|---|
| `klines` | bougies 1 s (spot et perpétuel) |
| `spot` / `perp` | un seul marché (`perp` inclut la référence 1 min et la vérification) |
| `reference` | bougies 1 min officielles |
| `funding` | funding rate |

**Mode de validation.** Par défaut (`VALIDATION_MODE=complete`), l'absence d'un dataset requis, d'un mois ou d'une référence fait échouer pytest. Pour explorer un sous-ensemble volontairement : `VALIDATION_MODE=partial pytest`. Un résultat partiel **ne vaut pas validation**.

**Ce que la vérification du perpétuel ne prouve pas.** C'est un contrôle par un pipeline de calcul indépendant, mais du même fournisseur : des lacunes communes aux deux produits ne seraient pas vues. Des trades proches d'un changement de minute sont parfois rangés dans la minute voisine chez Binance. La documentation de Binance indique que les aggTrades regroupent les trades de même prix et même côté sur 100 ms, et excluent ceux du fonds d'assurance et de l'ADL. Cela rend ces écarts plausibles, mais leur mécanisme exact et leur effet sur des features à la seconde **ne sont pas démontrés**. `python diagnose_perp.py` affiche les écarts mois par mois, avec les mêmes calculs et les mêmes seuils que pytest.

Les seuils se règlent par variables d'environnement (liste en tête de `tests/test_data_quality.py`, valeurs par défaut dans `quality_checks.py`).

---

## Points d'attention pour la suite

- **Entraîner sur le perpétuel**, le marché où le bot passera ses ordres.
- **Ne pas utiliser les prix bruts** comme features : utiliser des rendements et des variations relatives.
- **Normaliser les volumes** : ils varient beaucoup selon les périodes (par exemple, pas de frais sur les paires BTC du spot de mi-2022 à mars 2023).
- **Intégrer tous les coûts** dans les variables cibles et le backtest : frais (environ 0,05 % par ordre taker), funding et spread (inconnu : les données ne contiennent que des trades).
- **Découper train / test par période**, jamais au hasard.
- **Tester la causalité des features** : modifier le futur ne doit pas changer une feature passée ; le funding doit être disponible à l'instant de décision ; aucune fenêtre ni cible ne doit franchir un trou long ou une zone du registre ; la normalisation est ajustée sur l'entraînement seulement.
- **Convention du backtest** : quand un stop et un objectif sont touchés dans la même seconde, une bougie ne dit pas lequel est arrivé en premier. Cette convention devra être explicite.

---

## Feuille de route

- [x] Téléchargement incrémental des données (spot, perpétuel, funding)
- [x] Tests de qualité, registres d'exceptions, tests unitaires du pipeline
- [ ] Construction des features (et tests de causalité)
- [ ] Construction des trois variables cibles (entrée, direction, sortie)
- [ ] Serveur MLflow local et entraînement des modèles
- [ ] Backtest avec frais, funding et levier