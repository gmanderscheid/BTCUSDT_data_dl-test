# Journal de séance : constitution d'une donnée BTCUSDT de qualité

> Document destiné à un relecteur (humain ou LLM) qui découvre le dépôt. Il résume une séance de travail entre le porteur du projet et un assistant IA, et justifie les choix faits pour obtenir une donnée **propre, cohérente et traçable**, destinée à construire des features et à entraîner des modèles de trading.
>
> Date de la séance : 26/09/2026. Tous les horodatages cités sont en UTC.

---

# Partie 1 : Sommaire

## Objectif du projet

Bot de trading sur le **perpétuel BTCUSDT de Binance** (levier ×5 maximum), avec trois modèles : **quand entrer**, **quelle position prendre** (long / short) et **quand sortir**. Les expériences seront suivies avec un serveur MLflow local.

## Objectif de la séance

Récupérer **gratuitement** un historique **à la seconde depuis janvier 2020**, puis garantir sa qualité **avant** toute construction de features. Une donnée fausse ne fait pas planter le code : elle produit un modèle excellent en backtest et perdant en réel. La séance a donc été consacrée à trouver, expliquer et corriger les défauts de la donnée, pas à la modélisation.

## Ce qui a été produit

| Fichier | Rôle |
|---|---|
| `download_binance.py` | Téléchargement incrémental depuis Binance Vision, reconstruction des bougies 1 s du perpétuel, dédoublonnage, réparation des jours manquants |
| `gaps.py` | Détection des secondes manquantes, registre des trous longs, registre des minutes irréparables |
| `diagnose_perp.py` | Diagnostic mois par mois : bougies 1 s reconstruites vs bougies 1 min officielles |
| `tests/test_data_quality.py` | Environ 30 tests de qualité (pytest), étiquetés par catégorie et par marché |
| `pytest.ini` | Déclaration des étiquettes de tests |
| `data/known_gaps.csv` | Registre des trous longs (fermetures de l'exchange), versionné |
| `data/known_minute_mismatches.csv` | Registre des minutes irréparables et des zones invalides, versionné |
| `README.md`, `GLOSSAIRE.md` | Documentation d'utilisation et vocabulaire financier |
| `requirements.txt`, `.gitignore` | Dépendances et exclusions Git |

## Datasets obtenus

| Dataset | Contenu | Rôle |
|---|---|---|
| `futures_klines_1s` | Bougies 1 s du perpétuel, **reconstruites** à partir des aggTrades | Données de travail principales |
| `futures_klines_1m` | Bougies 1 min officielles du perpétuel | Référence indépendante pour vérifier la reconstruction |
| `futures_funding` | Funding rate (toutes les 8 h) | Coût de détention et feature |
| `spot_klines_1s` | Bougies 1 s du spot (publiées par Binance) | Feature complémentaire (écart spot / perpétuel) |

## Principaux défauts découverts et corrigés

1. **Journées entières absentes** des fichiers mensuels de Binance → réparées avec les fichiers journaliers.
2. **Journées tronquées** (par exemple le 13/04/2020 après 00:32) → réparées en croisant les bougies 1 s et 1 min.
3. **aggTrades en double** : 4,3 millions de doublons en septembre 2022 (volume exactement doublé les 12 et 13/09) → dédoublonnage par identifiant.
4. **Journées manquantes enregistrées à tort comme « fermetures de l'exchange »** dans le registre des trous → détecté par la vérification croisée, puis corrigé.
5. **Défauts dans les tests eux-mêmes** (sauts de prix comparés à travers des trous, contrôle de volume à sens unique, seuils calibrés pour le spot) → corrigés.

## État final

À la fin de la séance, tous les tests passaient, et les écarts résiduels avec la référence officielle étaient expliqués, quantifiés et enregistrés dans des registres versionnés.

**Une revue externe (Astra, 26/09/2026) a ensuite montré que « pytest vert » ne suffisait pas encore à valider l'historique** : couverture non bloquante, contrôles de prix et de volume trop permissifs, registre qui valait acceptation, absence de tests unitaires du pipeline. Tous ses constats ont été pris en compte (partie 3). Les tests doivent être relancés sur l'historique réel avec la nouvelle version.

## Limites connues, à garder en tête

- `n_trades` du perpétuel est une **approximation par excès**, à ne pas utiliser comme valeur exacte.
- Il n'y a **pas de données de carnet d'ordres** (ni L1, ni L2) : le spread devra être estimé dans le backtest.
- Deux explications restent des **hypothèses plausibles, non démontrées** (voir partie 2, section 6).
- Les minutes et zones des registres doivent être **exclues lors de la construction des features**.

---

# Partie 2 : Version détaillée

## 1. Choix de la source de données

**Source retenue : Binance Vision** (`data.binance.vision`), un dépôt public de fichiers ZIP mensuels et journaliers, gratuit et sans clé API.

**Pourquoi le perpétuel plutôt que le spot.** Le bot tradera des contrats perpétuels à levier. Leur prix est proche du spot mais pas identique : l'écart (*basis*) varie en permanence et s'élargit dans les phases agitées. Pour des trades de quelques secondes ou minutes, cet écart est du même ordre que les gains visés. Entraîner sur le spot et trader le perpétuel introduirait donc un biais. Décision : **entraîner et backtester sur le perpétuel**, et garder le spot comme feature.

**Pourquoi reconstruire les bougies 1 s.** Binance publie des bougies 1 s pour le spot, mais **pas pour les futures** (le plus petit intervalle est 1 min, vérifié sur le bucket S3 de Binance Vision). Les bougies 1 s du perpétuel sont donc reconstruites à partir des **aggTrades** (transactions agrégées), disponibles depuis janvier 2020.

**Nature de la donnée.** Ce sont des **transactions exécutées** (trades), ni du L1 ni du L2. On dispose du prix, du volume et du **sens de l'agresseur** (`taker_buy_base`), qui est une bonne mesure du flux d'ordres. On ne dispose ni du spread ni de la profondeur du carnet.

## 2. Principes de conception du téléchargement

| Principe | Mise en œuvre | Justification |
|---|---|---|
| Incrémental | Un Parquet par mois ; un mois présent n'est pas re-téléchargé | Reprise possible après interruption |
| Pas de doublons | Fusion dédoublonnée sur la clé temporelle (`merge_into`) | Relancer le script ne modifie pas les données existantes |
| Écriture atomique | Écriture dans un `.tmp`, puis renommage | Jamais de fichier à moitié écrit |
| Intégrité | Vérification SHA256 avec les `.CHECKSUM` de Binance | Détecte les téléchargements corrompus |
| Mémoire maîtrisée | ZIP écrit sur disque par blocs, aggTrades lus par morceaux de 5 millions de lignes | Un fichier de 700 Mo ne passe jamais en mémoire |
| Unités de temps | Détection ms / µs fichier par fichier | Binance est passé aux µs pour le spot en 2025 |
| Robustesse réseau | 3 tentatives par fichier | Téléchargement de plusieurs dizaines de Go |

**Reconstruction des bougies 1 s** (`build_bars_from_aggtrades`) : open / close = premier / dernier trade de la seconde dans l'ordre du fichier ; high / low = max / min ; volume = somme des quantités ; quote_volume = somme de prix × quantité ; volume acheteur = trades où `is_buyer_maker == False`. Une seconde coupée entre deux morceaux de lecture est ré-agrégée avec les mêmes règles. Ce découpage a été vérifié sur des données synthétiques, avec des frontières de morceaux forcées.

## 3. Tests de qualité : conception

Chaque test a une docstring qui explique **quelle erreur de modèle il évite**. Les tests sont étiquetés pour pouvoir être relancés par partie (`pytest -m <catégorie>`).

| Catégorie | Contrôles | Risque évité |
|---|---|---|
| `structure` | fichier non vide, colonnes, types, UTC, absence de NaN, pas de `.tmp` | colonnes décalées, prix lus comme texte, décalage horaire (fuite de données) |
| `chronologie` | doublons, tri, bon mois, alignement sur la seconde, durée des bougies | décalages de lignes qui faussent les variables cibles, frontières train / test brouillées |
| `completude` | trous courts / longs, mois manquants, fraîcheur, régularité du funding | marché immobile fabriqué par un comblement naïf, mois oublié |
| `coherence` | prix positifs, cohérence OHLC, sauts de prix, bornes du funding | points aberrants appris comme des signaux |
| `volumes` | volumes positifs, volume acheteur ≤ total, VWAP entre low et high | inversion de colonnes, feature de pression acheteuse faussée |
| `registre` | registre des trous bien formé | coupure des séries au mauvais endroit |
| `verification` | perpétuel 1 s agrégé vs bougies 1 min officielles | **erreur plausible mais fausse**, invisible aux tests de cohérence interne |

La catégorie `verification` est la plus importante : c'est la seule qui confronte la donnée à une **source indépendante**. C'est elle qui a révélé les défauts les plus graves.

## 4. Traitement des trous

Binance ne publie pas de bougie pour une seconde sans trade. Deux cas sont distingués :

| Cas | Critère | Cause | Traitement prévu dans les features |
|---|---|---|---|
| Trou court | ≤ 60 s | pas de trade pendant quelques secondes | reporter le dernier prix, volume à 0 |
| Trou long | > 60 s | exchange fermé (maintenance, panne) | **couper la série** : aucune fenêtre ni variable cible ne doit le traverser |

Les trous longs sont recensés dans `data/known_gaps.csv` par `python gaps.py`. Un trou long absent du registre fait échouer les tests. **Le registre doit être relu, et non accepté aveuglément** : voir la section 5.3.

**Bougies tronquées.** Quand Binance arrête le trading au milieu d'une seconde, la dernière bougie dure moins de 999 ms (parfois 0 ms). Constaté sur 8 dates du spot. 7 d'entre elles précèdent un trou long du registre, et la huitième (24/12/2021) un trou court. Règle adoptée : une bougie tronquée est acceptée **si et seulement si la seconde suivante est manquante**.

## 5. Chronologie des défauts trouvés

### 5.1 Spot : trous et maintenances

Sur le spot, les secondes manquantes de 2020–2021 formaient des plages aux durées rondes (1 h, 1 h 30, 2 h 30, 4 h 30…), typiques de **maintenances programmées**. À partir de 2023, il n'y a presque plus de trous. Ces trous sont réels et enregistrés dans le registre. Après calibrage, tous les tests du spot passent.

### 5.2 Perpétuel, premier passage : 246 échecs

Le premier lancement des tests sur le perpétuel a donné 246 échecs. Trois familles :
- trous courts très nombreux (jusqu'à 45 % des secondes en 2020) ;
- quelques sauts de prix supérieurs à 10 % ;
- comparaison aux bougies 1 min officielles en échec sur presque tous les mois.

### 5.3 Journées entières absentes des fichiers mensuels

**Constat :** des minutes manquantes par blocs d'exactement 1 440 (1 440, 2 880, 4 320), commençant à minuit. Ce sont des journées entières absentes des fichiers mensuels d'aggTrades.

**Défaut aggravant :** le test des trous longs passait, car `gaps.py` avait inscrit ces journées manquantes dans le registre comme de « vrais » trous. **Le registre ne protège que s'il est relu.**

**Correction :** `fill_missing_days` recherche chaque jour totalement absent dans le fichier journalier correspondant.

### 5.4 Journées tronquées

**Constat :** après la réparation précédente, il restait des journées partiellement absentes, par exemple le 13/04/2020 à partir de 00:32.

**Correction :** `repair_perp_partial_days` croise les deux datasets. Un jour où il nous manque des minutes officielles est re-téléchargé depuis le fichier journalier d'aggTrades. Un jour où il manque des minutes officielles est re-téléchargé depuis le fichier journalier 1 min. Les journées du 13/04, du 08/05 et du 08/11/2020 ont ainsi été complétées (48 000 à 77 000 secondes chacune).

### 5.5 Secondes vides du perpétuel

**Constat :** jusqu'à 45 % de secondes sans trade en 2020, et de 1 à 7 % ensuite.

**Analyse :** c'est réel. L'activité du perpétuel était faible en 2020, et les volumes concordent avec la référence officielle.

**Décision :** un seuil par marché. Le spot reste à 0,5 % ; le perpétuel est à 60 %, comme simple garde-fou. La vraie vérification de complétude du perpétuel est la catégorie `verification`.

### 5.6 Sauts de prix

- **06/11/2020 00:00:00** : artefact du test, qui comparait deux bougies séparées par plusieurs jours manquants. **Correction :** seules les secondes consécutives sont comparées.
- **18/04/2021 03:35:44** : +12 % en une seconde. La bougie 1 min officielle confirme le plus haut (58 839,06). **C'est une vraie mèche** du krach éclair de ce jour-là. **Décision :** seuil de 15 % pour le perpétuel, 10 % pour le spot. Ces mèches doivent rester dans les données : avec un levier ×5, elles déclenchent des liquidations, et le backtest doit simuler les stop-loss sur high et low.

### 5.7 Écarts minute par minute avec les bougies officielles

**Constat :** le volume **mensuel** est identique à l'officiel, mais 10 à 22 % des minutes diffèrent. Le close est identique dans 100 % des minutes ; l'open diffère dans 3 à 7 % des minutes.

**Première hypothèse, partiellement invalidée :** la documentation de Binance indique que les aggTrades excluent les trades du fonds d'assurance et de l'ADL. Nos trades formeraient donc un sous-ensemble, et on ne dépasserait jamais l'officiel. Or le test de bornes a montré qu'on le **dépasse** sur certaines minutes. Cette hypothèse ne suffit donc pas.

**Explication retenue, étayée par une mesure :** **76 % des minutes en excès sont compensées exactement par la minute voisine** (juin 2023). Certains trades situés près d'un changement de minute sont rangés dans la minute d'à côté chez Binance. *Reformulé après la revue* : la compensation des volumes montre qu'il s'agit de déplacements, mais ne borne ni leur amplitude, ni leur effet sur des features à la seconde (un trade proche d'une frontière peut changer de seconde). Leur effet n'est donc **pas démontré négligeable**.

**Refonte des tests de vérification** pour qu'ils restent stricts malgré ces déplacements :

| Test | Règle | Pourquoi elle tient |
|---|---|---|
| `test_no_missing_minutes` | aucune minute officielle absente chez nous | détecte les trades perdus |
| `test_prices_within_neighbours` | notre high / low reste dans celui de la minute officielle et de ses 2 voisines (tolérance 0,05 %) | un trade déplacé garde son prix |
| `test_no_volume_drift` | l'écart de volume **cumulé** ne dépasse pas 0,1 % du mois (volume et volume acheteur) | un trade déplacé se compense aussitôt ; un trade perdu ou en double crée une dérive qui persiste |
| `test_volume_deficit` | volume du mois concordant à 1 % près, **dans les deux sens** | contrôle d'ensemble |

Chaque règle a été validée sur des données synthétiques, dans les deux sens : 3 000 trades déplacés passent, alors que 30 % de trades perdus sur un jour, une inversion du sens acheteur/vendeur et un prix faussé de 1 % sont détectés.

**`n_trades`** : notre valeur, calculée à partir des plages d'identifiants des aggTrades, dépasse l'officielle même quand les volumes concordent. Ces plages incluent des identifiants qui ne sont pas des trades de marché. `n_trades` est donc retiré des comparaisons et documenté comme approximatif.

### 5.8 aggTrades en double (septembre 2022)

**Constat :** le test de dérive signalait 2 millions de BTC d'écart (9 % du mois). Le détail par jour a montré un volume **exactement +100,00 %** les 12 et 13/09/2022. Le fichier mensuel de Binance contenait ces journées en double. Il lui manquait aussi les journées du 01/09 et du 10/09.

**Correction :** dédoublonnage par `agg_trade_id` (unique et croissant). Seule la première occurrence est conservée, même si le doublon apparaît plus loin dans le fichier. Testé sur un fichier dont chaque ligne est dupliquée et sur un fichier avec un bloc répété à la fin : les bougies obtenues sont identiques à l'original. À la reconstruction réelle, **4 316 651 doublons** ont été retirés.

**Défaut de test associé :** le contrôle du volume mensuel ne vérifiait qu'un **déficit** et a donc laissé passer l'excédent. Il vérifie maintenant l'écart dans les deux sens.

### 5.9 Écarts irréparables : registre et zones invalides

Certaines minutes diffèrent même après re-téléchargement des fichiers journaliers (0 ligne ajoutée). Elles sont enregistrées dans `data/known_minute_mismatches.csv` par `python diagnose_perp.py --register`, avec leur volume officiel pour permettre le contrôle :

| Type (`side`) | Signification | Cas rencontrés |
|---|---|---|
| `absente_chez_nous` | bougie officielle, mais aucun aggTrade | 09/02/2021 (26 min), 19/05/2021 (24 min, jour de krach), 06/09/2022 17:15–17:19 (5 min, **2 000 à 2 900 BTC/min, donc de vrais trades absents des aggTrades**), 29/08/2025 (1 min) |
| `absente_chez_binance` | aggTrades présents, mais pas de bougie officielle | 10/11/2023, 28/10/2024 (74 min), 14/01/2025, 29/01/2025 |
| `zone_invalide` | ajoutée à la main : les deux côtés existent mais ne sont pas fiables | 06/09/2022 17:10–17:14 : prix dépassant l'officiel juste avant le trou de 17:15 (incident chez Binance) |

Ces minutes sont exclues des comparaisons. **Elles devront aussi être exclues de la construction des features.**

Deux écarts mineurs, acceptés par les tolérances : 2025-03-10 08:21 (notre high à 82 047,8 contre 82 032,3, soit +0,02 %), et 2026-06 (dérive de 0,07 % sur la première semaine, cohérente avec des trades de liquidation absents des aggTrades).

## 6. Hypothèses non démontrées

Pour la transparence envers le relecteur, voici les explications qui restent des **hypothèses plausibles** :

1. **Déplacement de trades entre minutes voisines.** Il est étayé par la mesure (76 % de compensation exacte, volume mensuel identique). En revanche, le mécanisme précis (horodatage des aggTrades vs horodatage des bougies officielles) n'est pas confirmé par une source Binance.
2. **Minutes « absentes chez nous » dues aux liquidations.** C'est plausible pour les jours de krach (19/05/2021), puisque la documentation de Binance confirme que les aggTrades excluent les trades du fonds d'assurance et de l'ADL. Ce n'est **pas** le cas du 06/09/2022, où le volume manquant est un volume de marché normal : c'est un trou dans la source.
3. **Mêmes volumes mensuels malgré des trades de liquidation exclus.** Le volume mensuel est identique à 0,00 % sur la plupart des mois. Cela suggère que les bougies officielles excluent elles aussi ces trades, ou qu'ils sont négligeables. Ce point n'a pas été tranché.

## 7. Erreurs de l'assistant pendant la séance

Ces erreurs sont listées pour que le relecteur mesure la fiabilité du processus. Chacune a été détectée par les tests ou par l'analyse des résultats, puis corrigée.

| Erreur | Conséquence | Correction |
|---|---|---|
| Affirmation que les klines 1 s du spot n'existaient que depuis 2022 | Aucune : les données de 2020 ont été téléchargées | Rectifiée au vu des fichiers réels |
| Hypothèse d'une bougie « trop longue » après les maintenances | Mauvais diagnostic initial | Les bougies étaient trop **courtes** (arrêt en cours de seconde) ; règle corrigée |
| Test des sauts de prix à travers les trous | Faux positif (06/11/2020) | Comparaison limitée aux secondes consécutives |
| Hypothèse du « sous-ensemble strict » pour la vérification | Test de bornes faux sur tous les mois | Remplacé par les tests de voisinage et de dérive |
| Test par fenêtres croissantes, annoncé comme discriminant | Mauvaise conclusion possible | Reconnu comme non concluant : une fenêtre a toujours 2 frontières |
| Contrôle de volume mensuel à sens unique | Excédent de septembre 2022 non détecté par ce test | Contrôle dans les deux sens |
| Exclusion des minutes du registre sans leurs voisines | Faux positifs de bord (28/10/2024 19:59) | Exclusion des minutes voisines d'un trou officiel enregistré |

## 8. Seuils et paramètres

Tous les seuils se règlent par variable d'environnement. Valeurs par défaut :

| Variable | Défaut | Justification |
|---|---|---|
| `SHORT_GAP_MAX_SECONDS` | 60 | limite entre secondes sans trade et fermeture de l'exchange |
| `MAX_SHORT_GAP_RATIO` | 0,5 % | spot : klines 1 s presque complètes |
| `MAX_SHORT_GAP_RATIO_PERP` | 60 % | perpétuel : secondes vides réelles, simple garde-fou |
| `MAX_PRICE_JUMP` / `_PERP` | 10 % / 15 % | mèche réelle de +12 % sur le perpétuel le 18/04/2021 |
| `MAX_PRICE_EXCESS` | 0,05 % | trades isolés légèrement hors de l'extrême officiel (+0,02 % constaté) |
| `MAX_VOLUME_DRIFT` | 0,1 % du mois | +0,07 % constaté en juin 2026 ; 9 % pour les doublons de 2022 |
| `MAX_VOLUME_DEFICIT` | 1 % | contrôle d'ensemble, dans les deux sens |

## 9. Procédure de reproduction

```bash
pip install -r requirements.txt
python download_binance.py          # téléchargement + réparations (plusieurs heures la première fois)
python gaps.py                      # registre des trous longs : À RELIRE
python diagnose_perp.py --register  # minutes irréparables : À RELIRE (volume officiel affiché)
pytest                              # tous les tests doivent passer
```

## 10. Points d'attention pour le relecteur

Suggestions de points à examiner dans le code :

1. **`build_bars_from_aggtrades`** : ordre de first / last lors de la ré-agrégation entre morceaux ; dédoublonnage par `agg_trade_id`, qui suppose des identifiants croissants dans le fichier.
2. **`merge_into`** : le dédoublonnage porte sur `open_time`. Une seconde déjà présente mais incomplète (fichier tronqué au milieu d'une seconde) ne serait pas complétée par une réparation.
3. **`repair_perp_partial_days`** : retente à chaque lancement les jours irréparables déjà enregistrés. C'est sans effet sur la donnée, mais ça coûte quelques secondes.
4. **`to_datetime`** : la détection ms / µs se fait par seuil sur la valeur maximale de chaque fichier ou morceau.
5. **Tests de vérification** : logique d'exclusion des minutes du registre et de leurs voisines ; calcul de la dérive cumulée.
6. **Seuils** : pertinence des valeurs par défaut (section 8).
7. ~~Absence de test unitaire du code lui-même.~~ Corrigé après la revue : voir `tests/test_pipeline.py` (partie 3).

## 11. Suite prévue

La donnée étant validée, les prochaines étapes sont :
- la construction des features (rendements plutôt que prix bruts, volumes normalisés, séries coupées aux trous et aux zones invalides) ;
- les trois variables cibles, en intégrant les frais (environ 0,05 % en taker), le funding et une estimation du spread (méthode *triple barrier* envisagée) ;
- le suivi des expériences dans MLflow ;
- un backtest avec levier, qui simule les stop-loss sur high et low.

---

# Partie 3 : Prise en compte de la revue externe

La revue d'Astra (26/09/2026) a examiné le code et exécuté des contre-exemples sur données synthétiques. Son avis : une base utile, mais pas encore suffisante pour considérer l'historique comme « validé pour l'entraînement ». Ses constats étaient fondés : plusieurs contre-exemples passaient les tests alors qu'ils auraient dû échouer. Chacun a été corrigé, et **chaque contre-exemple est désormais un test unitaire** dans `tests/test_pipeline.py`. Il échouerait si la correction était retirée.

## Changements d'architecture

| Nouveau fichier | Rôle |
|---|---|
| `quality_checks.py` | Tous les contrôles et seuils, partagés par pytest et `diagnose_perp.py` (plus de divergence entre diagnostic et validation) |
| `data_contract.py` | Contrat de couverture : datasets requis, premier mois (2020-01), dernier mois attendu, référence 1 min pour chaque mois du perpétuel |
| `approve.py` | Approbation tracée des anomalies examinées |
| `tests/test_pipeline.py` | 61 tests unitaires sur données synthétiques, sans données réelles |
| `data/manifest.csv`, `data/repair_attempts.csv` | Provenance de chaque fichier intégré, réparations déjà tentées |

## Traçabilité constat → correction → preuve

| # | Constat de la revue | Correction | Test qui le prouve (`test_pipeline.py` sauf mention) |
|---|---|---|---|
| 1 | Sans données ou avec une référence absente, pytest reste vert (35 tests ignorés) | Contrat de couverture bloquant ; mode `partial` explicite et signalé ; part minimale de minutes comparées (99 %) ; volume exclu par le registre borné (0,5 %) ; minutes absentes de la référence bloquantes | `TestCoverageContract`, `test_comparison_coverage` (données) ; vérifié : sans données, 16 échecs en mode `complete` |
| 2 | OHLC plats à 100 contre 99/110/90/101 : la vérification passe | Détection des **mèches disparues** (extrême officiel absent de nos données) en plus des extrêmes inventés ; contrôle du close (≤ 0,5 % de minutes différentes) ; reconstruction vérifiée contre des valeurs calculées à la main | `test_flat_bars_with_missing_wicks_fail`, `TestReconstruction.test_exact_bars` (tailles de morceaux 1, 2, 3, 100 ; avec et sans en-tête ; ms et µs) |
| 3 | Une heure amputée de 30 % passe (0,04 % du mois) | Contrôles **locaux** : volume et volume acheteur par jour (0,5 %) et par heure (5 % ou 100 BTC) ; `test_volume_deficit` documenté comme portant sur les minutes communes | `test_one_hour_loss_is_caught_locally`, avec le cas positif `test_shifted_trades_are_accepted` |
| 4 | Voisines = lignes : 00:00 / 00:01 / 02:00 élargit la référence | Voisines = minutes **exactes** t−1, t, t+1 ; une minute dont une voisine n'existe que d'un côté est non vérifiable (et comptée) | `test_neighbours_are_exact_minutes` |
| 5 | Un seul paiement de funding passe les 6 tests | **Échéancier exact** du mois (tous les multiples de l'intervalle depuis 00:00 : ni manquant, ni en trop) ; schéma et UTC ; médiane du taux contre une erreur d'unité ×100 | `TestFundingChecks` (paiement manquant au début, à la fin, au milieu ; fichier à une ligne ; paiement en trop ; erreur d'unité) |
| 6 | Le registre vaut acceptation (raison vide acceptée) | Statuts `candidate` / `approved` ; raison obligatoire ; `approve.py` ; exceptions obsolètes bloquantes ; test de structure du registre des minutes | `TestRegistryChecks`, `TestGapRegistry`, `TestMismatchRegistry` (données) |
| 7a | Durée négative acceptée avant un trou | `close_time >= open_time` imposé ; seconde suivante cherchée aussi dans le fichier du mois suivant ; fin en µs acceptée | `test_negative_duration_before_gap_fails`, `test_truncated_candle_rules`, `test_microsecond_end_is_normal` |
| 7b | `n_trades = +inf` passe ; référence 1 min sans contrôles propres | Contrôle de finitude et `n_trades` entier ; classe `TestReference1m` (structure, temps, valeurs) ; dérive NaN bloquante | `test_infinite_n_trades_fails`, `test_nan_drift_fails` |
| 7c | Seuil de saut logarithmique : −9,5 % échoue à « 10 % » | Rendement **simple**, symétrique | `test_price_jump_is_symmetric_in_simple_return` |
| 7d | Marge VWAP non documentée ; pas de VWAP acheteur | Marge documentée comme non calibrée ; VWAP acheteur ; cohérence des volumes nuls | `test_buyer_vwap_out_of_range_fails`, `test_quote_without_base_fails` |
| 8a | Checksum en erreur 500 : ZIP accepté | Checksum **obligatoire** : faux, vide, mal formé ou erreur serveur = échec (après nouvelles tentatives) ; absence de `.CHECKSUM` bloquante sauf `--allow-missing-checksum` | `TestDownload` (6 cas) |
| 8b | `--force` ne remplace pas une valeur existante | `--force` **remplace** le mois ; réparation par remplacement de la journée si le fichier journalier couvre au moins les mêmes instants ; jamais d'addition de deux bougies d'une même seconde | `test_force_replaces_existing_month`, `test_replace_only_if_superset`, `test_merge_is_idempotent_and_add_only` |
| 8c | Journée amputée sans minute disparue non réparée | Réparation déclenchée aussi par un écart de volume journalier > 0,5 % avec la référence | `test_day_with_all_minutes_but_missing_volume_is_flagged` |
| 8d | Identifiants 1, 3, 2, 4 : trade inédit supprimé silencieusement | Ordre **vérifié** : identifiant inédit hors ordre = `AggTradeOrderError` ; identifiants vus stockés en intervalles (mémoire compacte) ; horodatages en recul signalés | `test_unordered_new_id_fails_loudly` (morceaux de 2 et 10), `test_repeated_block_removed`, `test_id_ranges` |
| 9a | Trou coupé par minuit (40 s + 50 s) jamais « long » | Trous **fusionnés** aux changements de mois avant classement | `test_gap_split_by_month_change_is_merged` |
| 9b | `diagnose_perp.py` et pytest divergent | Mêmes fonctions et mêmes seuils (`quality_checks.py`) ; vue brute et vue filtrée distinguées ; détail des échecs pytest par mois | — (architecture) |
| 9c | `-m perp` n'exécute pas la couverture du perpétuel | Étiquettes de dataset sur les cas de couverture et les registres | — |
| 9d | Versions minimales seulement, pas de provenance | `data/manifest.csv` (source, SHA256, date, mode, empreinte du code) ; `pip freeze > requirements.lock` documenté | — |
| — | Hypothèses trop affirmatives | « Sans conséquence » retiré ; mention de l'agrégation sur 100 ms de la documentation Binance ; mécanisme exact déclaré non démontré (README, docstrings, section 5.7) | — |

## Vérifications effectuées

- **61 tests unitaires** passent, dont les 12 contre-exemples de la revue, transformés en tests qui échouent si la correction est retirée.
- **Sans données**, en mode `complete` : 16 échecs de couverture. Le constat n°1 est corrigé.
- **Sur une simulation de 3 mois** (Binance Vision simulé en local, avec journée manquante, trades exclus, minutes irréparables) : téléchargement, réparation, manifeste et registres fonctionnent de bout en bout. Les anomalies bloquent pytest tant qu'elles ne sont pas approuvées, et pytest passe une fois qu'elles le sont.

## Ce qui reste à faire sur l'historique réel

1. `python download_binance.py`. Les journées dont le volume diffère de la référence seront re-téléchargées.
2. `python gaps.py`, puis `python diagnose_perp.py --register`. Toutes les anomalies existantes redeviennent **candidates**, puisque les anciens registres n'avaient pas de statut.
3. Examiner chaque journée (`python approve.py --list`) et l'approuver avec une raison qui dit ce qui a été vérifié.
4. `pytest`. Les nouveaux contrôles (mèches disparues, volumes par heure, échéancier du funding, close) n'ont **jamais été exécutés sur les données réelles**. Des échecs sont possibles : ils devront être analysés, et non contournés en relâchant les seuils uniformément.
5. `pip freeze > requirements.lock` au moment de la validation.

## Limites qui demeurent

- La référence 1 min vient du même fournisseur que les aggTrades : une lacune commune aux deux ne serait pas vue.
- Une concordance à la minute ne prouve pas l'exactitude seconde par seconde. Cette exactitude repose sur les tests unitaires de la reconstruction.
- Les seuils (hors valeurs constatées pendant la séance) n'ont pas été calibrés sur l'historique complet. Il faut mesurer ce qu'ils laissent passer avant de les modifier.
- Les tests de **causalité** des futures features, et la convention du backtest quand un stop et un objectif sont touchés dans la même seconde, restent à écrire (README, « Points d'attention »).