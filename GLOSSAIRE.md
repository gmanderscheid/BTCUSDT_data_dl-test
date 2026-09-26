# Glossaire

Les termes de finance et de trading utilisés dans le code, les noms de fichiers et les tests du projet. Les termes sont regroupés par thème, et le nom anglais utilisé dans le code est indiqué entre parenthèses quand il diffère.

---

## 1. Marchés et produits

**Exchange (plateforme d'échange)**
Entreprise qui met en relation acheteurs et vendeurs de cryptomonnaies : Binance, Bybit, Coinbase… Elle tient le carnet d'ordres, exécute les transactions et publie les données de marché.

**Paire (symbol)**
Les deux actifs échangés l'un contre l'autre. `BTCUSDT` signifie « bitcoin coté en USDT » : le prix indique combien d'USDT il faut pour acheter 1 BTC.
- L'**actif de base** (*base asset*) est le premier, BTC : c'est ce qu'on achète ou vend.
- L'**actif de cotation** (*quote asset*) est le second, USDT : c'est la monnaie dans laquelle le prix est exprimé.

**USDT (Tether)**
*Stablecoin*, c'est-à-dire une cryptomonnaie conçue pour valoir toujours 1 dollar américain. Sur les exchanges crypto, il sert de monnaie de référence à la place du dollar.

**Spot (marché au comptant)**
Le marché où l'on achète et vend l'actif réel, livré immédiatement. Acheter 1 BTC sur le spot, c'est posséder 1 BTC. On ne peut que parier à la hausse (acheter puis revendre plus cher), et sans levier.
Dans le code : `spot_klines_1s`, `data/spot/…`.

**Future (contrat à terme)**
Contrat qui permet de gagner ou perdre de l'argent sur la variation du prix du BTC, sans jamais posséder de BTC. On peut parier à la hausse comme à la baisse, et utiliser un levier. Un future classique a une date d'échéance à laquelle il est réglé.

**Perpétuel (perpetual future, « perp »)**
Future sans date d'échéance : on peut garder la position aussi longtemps qu'on veut. C'est le produit le plus utilisé en crypto, et celui que le bot tradera. Pour que son prix reste collé à celui du spot, il utilise le mécanisme de *funding* (voir plus bas).

**USD-M / COIN-M**
Les deux familles de futures chez Binance.
- **USD-M** (*USDⓈ-margined*) : la marge, les gains et les pertes sont en USDT. C'est le cas de BTCUSDT, et c'est le plus simple.
- **COIN-M** (*coin-margined*) : la marge et les gains sont en BTC.
Dans le code : `futures/um/…` désigne USD-M.

---

## 2. Positions et levier

**Position**
Ce que tu détiens à un instant donné : ton exposition au prix. Ouvrir une position, c'est entrer sur le marché ; la fermer (ou la « clôturer »), c'est en sortir et encaisser le gain ou la perte.

**Long**
Position qui gagne quand le prix monte. « Être long » = avoir acheté.

**Short (vente à découvert)**
Position qui gagne quand le prix baisse. Sur un future, on « vend » un contrat qu'on ne possède pas, puis on le rachète plus tard ; si le prix a baissé entre-temps, on empoche la différence.

**Levier (leverage)**
Multiplicateur entre ta mise et la taille de ta position. Avec 1 000 € et un levier ×5, tu contrôles une position de 5 000 €. Gains **et** pertes sont multipliés par 5 : une baisse de 2 % du prix fait perdre 10 % de ta mise.

**Marge (margin)**
L'argent que tu immobilises pour ouvrir une position à levier : ta mise. Dans l'exemple ci-dessus, la marge est de 1 000 €.

**Notionnel (notional)**
La taille réelle de la position : marge × levier, soit 5 000 € dans l'exemple. Les frais et le funding sont calculés sur le notionnel, pas sur la marge.

**Liquidation**
Fermeture forcée de ta position par l'exchange quand tes pertes approchent le montant de ta marge. Avec un levier ×5, une variation d'environ 20 % contre toi (un peu moins en pratique, à cause des frais et de la marge de maintenance) fait perdre toute la mise. C'est le risque principal du trading à levier.

**Stop-loss / take-profit**
Ordres de sortie automatiques : le *stop-loss* ferme la position si la perte atteint un seuil, le *take-profit* la ferme si le gain atteint un objectif. Ils seront au cœur de ta variable cible « quand sortir ».

---

## 3. Ordres et exécution

**Trade (transaction)**
Un échange effectif entre un acheteur et un vendeur : une quantité, un prix, un instant. C'est la donnée la plus élémentaire du marché.
Dans le code : `trades`, `n_trades` (nombre de trades dans la seconde).

**AggTrade (trade agrégé)**
Regroupement par Binance des trades exécutés au même instant, au même prix et du même côté (souvent un gros ordre qui en a rencontré plusieurs petits). Plus léger que les trades bruts, presque aussi informatif. Utile pour reconstruire des bougies 1s sur les futures.

**Carnet d'ordres (order book)**
Liste de tous les ordres en attente : les ordres d'achat (*bids*) d'un côté, les ordres de vente (*asks*) de l'autre, chacun avec son prix et sa quantité.
Dans le code Binance : `bookDepth`, `bookTicker`.

**Bid / Ask**
- *Bid* : le meilleur prix auquel quelqu'un accepte d'acheter en ce moment.
- *Ask* : le meilleur prix auquel quelqu'un accepte de vendre.

**Spread**
L'écart entre ask et bid. Si tu achètes puis revends immédiatement, tu perds le spread. Sur BTCUSDT, il est minuscule, mais à l'échelle de la seconde il compte.

**Maker / taker**
- Le **maker** place un ordre à prix fixé qui attend dans le carnet : il « fait » la liquidité.
- Le **taker** exécute immédiatement contre un ordre déjà présent : il « prend » la liquidité.
Le taker paie des frais plus élevés. Dans un trade, le taker est celui qui a pris l'initiative : c'est lui qui révèle la pression du marché.

**Taker buy volume**
Volume des trades où l'acheteur était le taker, c'est-à-dire où quelqu'un a acheté « au marché » de façon agressive. Rapporté au volume total, il mesure la pression acheteuse : proche de 1, les acheteurs dominent ; proche de 0, les vendeurs dominent.
Dans le code : `taker_buy_base` (en BTC), `taker_buy_quote` (en USDT).

**Slippage (glissement)**
Différence entre le prix attendu et le prix réellement obtenu, parce que le marché a bougé pendant l'envoi de l'ordre ou parce que l'ordre était trop gros pour le carnet.

**Frais (fees)**
Commission prélevée par l'exchange sur chaque trade, en pourcentage du notionnel (de l'ordre de 0,02 % en maker et 0,05 % en taker sur les futures Binance, variable selon le compte). Un aller-retour (entrée + sortie) coûte donc environ 0,1 % : à l'échelle de la seconde, c'est souvent plus que le mouvement de prix espéré.

---

## 4. Bougies et prix

**Bougie (kline, candlestick, chandelier)**
Résumé de tous les trades d'une période fixe (ici 1 seconde) en quelques chiffres : ouverture, plus haut, plus bas, clôture, volume. *Kline* est le terme utilisé par Binance.
Dans le code : `klines`, `spot_klines_1s` (bougies d'une seconde du spot).

**OHLC / OHLCV**
Les composantes d'une bougie :
- **Open** (`open`) : prix du premier trade de la période ;
- **High** (`high`) : prix le plus haut atteint ;
- **Low** (`low`) : prix le plus bas atteint ;
- **Close** (`close`) : prix du dernier trade ;
- **Volume** (`volume`) : le V de OHLCV (voir ci-dessous).

**open_time / close_time**
Début et fin de la période couverte par la bougie. Pour une bougie 1s : `close_time = open_time + 999 ms`.

**Intervalle (interval)**
La durée d'une bougie : `1s`, `1m` (minute), `1h`, `1d`… Plus l'intervalle est court, plus les données sont fines et volumineuses.

**Volume**
Quantité d'actif échangée pendant la période.
- `volume` : en actif de base (BTC).
- `quote_volume` : en actif de cotation (USDT), c'est-à-dire le montant d'argent échangé.

**VWAP (Volume-Weighted Average Price)**
Prix moyen pondéré par les volumes : `quote_volume / volume`. C'est le prix moyen réellement payé par les acheteurs de la période. Il est forcément compris entre `low` et `high`.

**Rendement (return)**
Variation relative du prix : `(prix_t / prix_{t-1}) - 1`. On utilise souvent le **rendement logarithmique** `log(prix_t / prix_{t-1})`, qui s'additionne facilement dans le temps et se comporte mieux statistiquement.

**Volatilité**
Amplitude des variations de prix, souvent mesurée par l'écart-type des rendements. Forte volatilité = prix qui bouge beaucoup = plus d'opportunités et plus de risque.

**Liquidité**
Facilité à acheter ou vendre une quantité donnée sans faire bouger le prix. Un marché liquide a beaucoup de trades, un spread faible et un carnet d'ordres épais.

---

## 5. Spécifique aux perpétuels

**Funding rate (taux de financement)**
Paiement périodique (toutes les 8 h pour BTCUSDT) entre longs et shorts, qui maintient le prix du perpétuel proche du spot.
- Taux positif : le perpétuel est plus cher que le spot ; les longs paient les shorts.
- Taux négatif : l'inverse, les shorts paient les longs.
On ne paie ou ne reçoit que si la position est ouverte à l'instant du prélèvement, et le montant est calculé sur le notionnel.
Dans le code : `futures_funding`, `last_funding_rate`, `calc_time` (instant du prélèvement), `funding_interval_hours`.

**Index price (prix de l'indice)**
Moyenne du prix spot du BTC sur plusieurs exchanges. C'est la référence « juste » que le perpétuel est censé suivre.
Dans le code Binance : `indexPriceKlines`.

**Mark price (prix de marque)**
Prix calculé par Binance à partir de l'index, qui sert à déterminer les liquidations. Il est moins manipulable que le dernier prix échangé, ce qui évite des liquidations provoquées par un pic isolé.
Dans le code Binance : `markPriceKlines`.

**Premium (prime)**
Écart entre le prix du perpétuel et l'index. C'est à partir de lui que le funding rate est calculé.
Dans le code Binance : `premiumIndexKlines`.

**Open interest (intérêt ouvert)**
Nombre total de contrats futures actuellement ouverts. S'il monte, de l'argent nouveau entre sur le marché ; s'il baisse, des positions se ferment.
Dans le code Binance : colonne `sum_open_interest` des fichiers `metrics`.

**Ratio long/short**
Proportion des comptes (ou des positions) longs par rapport aux shorts. Indicateur de sentiment : un ratio extrême signale souvent un marché déséquilibré, vulnérable à un retournement.

---

## 6. Données et fonctionnement de l'exchange

**Binance Vision**
Site de Binance (data.binance.vision) qui publie gratuitement l'historique des données de marché, en fichiers ZIP mensuels et journaliers.

**Timestamp (horodatage)**
Instant exprimé en nombre de millisecondes (ms) ou microsecondes (µs) écoulées depuis le 1er janvier 1970 UTC. Binance est passé des ms aux µs pour le spot à partir de 2025.

**UTC**
Temps universel coordonné, le fuseau horaire de référence. Toutes les données Binance sont en UTC ; Paris est à UTC+1 en hiver et UTC+2 en été.

**Maintenance**
Période où l'exchange est volontairement fermé pour une mise à jour technique. Aucun trade n'a lieu, ce qui crée des trous de plusieurs minutes ou heures dans les données.
Dans le code : les **trous longs** recensés dans `data/known_gaps.csv`.

**Trou (gap)**
Suite de secondes sans bougie. Un **trou court** correspond à des secondes sans trade (on reporte le prix précédent) ; un **trou long** correspond à une fermeture de l'exchange (on coupe la série). Voir `gaps.py`.

**Checksum (SHA256)**
Empreinte numérique d'un fichier. Binance publie celle de chaque ZIP : si l'empreinte du fichier téléchargé est identique, le fichier est intact.

---

## 7. Stratégie et évaluation

**Backtest**
Simulation d'une stratégie sur des données passées, pour estimer ce qu'elle aurait gagné ou perdu. Un backtest doit intégrer les frais, le spread, le slippage et le funding, sinon il est trop optimiste.

**Fuite de données (look-ahead bias)**
Erreur où le modèle utilise, sans le savoir, une information du futur pour prendre une décision dans le passé. Le backtest paraît excellent et la stratégie échoue en réel. Causes fréquentes : données mal triées, décalages de fuseau horaire, features calculées sur toute la série.

**Triple barrier (triple barrière)**
Méthode d'étiquetage des données (Marcos López de Prado) : pour chaque point d'entrée, on regarde lequel de trois événements arrive en premier : le take-profit (barrière haute), le stop-loss (barrière basse) ou une limite de temps (barrière verticale). Point de départ naturel pour tes variables cibles.

**Régime de marché**
Phase durable du marché aux caractéristiques propres : tendance haussière, baissière, marché calme, forte volatilité… Un modèle entraîné sur un seul régime peut échouer quand le régime change.