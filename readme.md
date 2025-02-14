# README - Travaux Pratiques : Butinages d'arbres urbains dans la métropole rennaise

## Contexte
Ce projet consiste à exploiter un jeu de données représentant les regroupements d'arbres sur le territoire de Rennes. L'objectif est d'appliquer des concepts d'algorithmes, de structures de données et de complexité afin de manipuler, visualiser et analyser ces données.

Données disponibles : [Jeu de données](https://data.rennesmetropole.fr/explore/dataset/regroupements_arbres/information/)

## Objectifs
Le projet est divisé en plusieurs étapes :

### 1. Manipulation de Données JSON
- Charger et transformer les données en liste de dictionnaires Python.
- Structurer les données en ne gardant que les informations utiles : latitude, longitude, dénomination, date de plantation, genre et ID unique.

### 2. Visualisation des Arbres sur une Carte
- Utiliser **Folium** pour afficher les arbres sous forme de marqueurs.
- Ajouter des infobulles affichant le genre et la date de plantation des arbres.

### 3. Calcul de Distances Géographiques
- Implémenter la **formule de Haversine** pour mesurer la distance entre deux arbres.
- Identifier les paires d'arbres les plus proches et les plus éloignés.

### 4. Construction et Analyse de Graphes
- Construire un **graphe complet** où chaque arbre est un nœud et chaque arête est une distance entre arbres.
- Extraire un sous-graphe où les arbres distants de moins de **200m** sont connectés.
- Identifier les arbres avec le plus grand nombre de voisins.

### 5. Recherche de Plus Courts Chemins
- Implémenter des algorithmes de recherche de **plus court chemin** pour optimiser le parcours des abeilles.
- Comparer **Dijkstra, A*, Bellman-Ford, et Floyd-Warshall**.

### 6. Étude des Arbres Mellifères
- Identifier les arbres favorables aux abeilles et simuler un scénario de collecte de pollen.

### 7. Optimisation du Placement des Ruches
- Déterminer la meilleure répartition des ruches pour maximiser la collecte de pollen et couvrir les arbres mellifères.

## Prérequis
### Outils et Bibliothèques Nécessaires
- **Python 3.x** (installation recommandée via un environnement virtuel)
- **Bibliothèques Python**:
  ```bash
  pip install -r requirements.txt
  ```
  - `folium`
  - `matplotlib`
  - `networkx`
  - `numpy`
  - `geopy`

## Livrables
Chaque groupe doit fournir :
- Un **notebook Python** avec le code et les analyses.
- Un **fichier requirements.txt** listant les bibliothèques nécessaires.
- Les **cartes générées** sous forme de fichiers HTML.
- Une **archive ZIP** contenant tous les fichiers (hors environnement virtuel).


