"""
source:
Json:
https://www.docstring.fr/glossaire/json/

Folium:
https://python-visualization.github.io/folium/latest/getting_started.html

Authors:

Armand COiffe
Matéo

"""

import json
import folium
from collections import deque


def chargement() -> list:
    """
    Charge les données depuis un fichier JSON et retourne une liste de dictionnaires.
    """
    with open("regroupements_arbres.json", "r") as f:
        data = json.load(f)
        return list(data)


def structuration() -> list:
    """
    Transforme les données pour ajouter un identifiant unique et conserver les champs importants.

    Returns:
        List[dict]: Liste des arbres structurés avec id, geo_point_2d, denomination, date_plantation, genre.
    """
    data = chargement()
    data_transformé = []
    for i in range(len(data)):
        arbre_i = {
            "id": i,
            "geo_point_2d": data[i]["geo_point_2d"],
            "denomination": data[i]["denomination"],
            "date_plantation": data[i]["date_plantation"],
            "genre": data[i]["genre"],
        }

        data_transformé.append(arbre_i)
    return data_transformé


"""
question 3.
print(structuration()[:5])

{'id': 1474, 'geo_point_2d': {'lon': -1.6683522129258248, 'lat': 48.08934977785336}, 'denomination': 'Rue Roger Chevrel', 'date_plantation': 2005, 'genre': 'Prunus'}
"""


def carte(D):
    """
    Génère une carte Folium avec les arbres marqués.

    Args:
        arbres (List[dict]): Liste des arbres structurés.
        output_file (str): Nom du fichier de sortie HTML pour la carte.
    """
    m = folium.Map(location=(48.083328, -1.68333), zoom_start=12)

    for e in D:
        folium.Marker(
            location=[e["geo_point_2d"]["lat"], e["geo_point_2d"]["lon"]],
            tooltip=e["denomination"],
            popup=f"data: {e['date_plantation']} \n genre: {e['genre']}",
            icon=folium.Icon(color="green"),
        ).add_to(m)
    m.save("carte.html")


"""
formule de Haversine:
http://villemin.gerard.free.fr/aGeograp/Distance.htm#haver
"""
import numpy as np


def harversine(lat1, lat2, lon1, lon2):
    """
    Calcule la distance entre deux points géographiques en mètres à l'aide de la formule de Haversine.

    Args:
        lat1, lat2, lon1, lon2 (float): Latitude et longitude des deux points en degrés.

    Returns:
        float: Distance entre les deux points en mètres.
    """
    lat1, lat2, lon1, lon2 = (
        np.radians(lat1),
        np.radians(lat2),
        np.radians(lon1),
        np.radians(lon2),
    )
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c * 1000  # rayon terre


def find_max_n_min():
    """
    Trouve les distances maximale et minimale entre tous les arbres.

    Returns:
        Tuple[float, float]: Distance maximale et minimale en mètres.
    """
    d_max = 0
    d_min = 0
    D = {}
    Arbres = structuration()
    for arbre1 in Arbres:
        for arbre2 in Arbres:
            if arbre1 != arbre2:
                D[(arbre1["id"], arbre2["id"])] = harversine(
                    arbre1["geo_point_2d"]["lat"],
                    arbre2["geo_point_2d"]["lat"],
                    arbre1["geo_point_2d"]["lon"],
                    arbre2["geo_point_2d"]["lon"],
                )

    D = dict(sorted(D.items(), key=lambda item: item[1]))
    d_min = next(
        iter(D.items())
    )  # next est l'équivalent de L[0] chez les dictionnaires
    d_max = next(reversed(D.items()))  # on utilise reverse pour avoir L[-1]
    return (d_max[0], d_min[0])


"""
print(find_max_n_min())


Question 9

disctionaire {arbre1: (disctance_arbre_2, arbre2)}
"""


def init_graph() -> dict:
    """
    Crée un graphe représentant les distances entre chaque paire d'arbres.

    Returns:
        Dict[int, List[Tuple[float, int]]]: Graphe où les clés sont les IDs des arbres et les valeurs
        sont des listes de tuples contenant la distance et l'ID des voisins.
    """
    Arbres = structuration()
    graphe = {}
    for arbre1 in Arbres:
        graphe[arbre1["id"]] = []
        for arbre2 in Arbres:
            if arbre1 != arbre2:
                graphe[arbre1["id"]].append(
                    (
                        harversine(
                            arbre1["geo_point_2d"]["lat"],
                            arbre2["geo_point_2d"]["lat"],
                            arbre1["geo_point_2d"]["lon"],
                            arbre2["geo_point_2d"]["lon"],
                        ),
                        arbre2["id"],
                    )
                )
    return Arbres, graphe


def verification(G: dict):
    """
    comptage
    """
    nombre_noeuds = len(structuration()[1])
    nombre_arretes = nombre_noeuds * (nombre_noeuds - 1)
    nombre_noeuds_verif = len(G)
    nombre_arretes_verif = 0
    for e in G.values():
        nombre_arretes_verif += len(e)
    return (
        nombre_noeuds == nombre_noeuds_verif,
        nombre_arretes == nombre_arretes_verif,
    )


"""
exercice 11:

print(verification(init_graph()[1]))

"""


def extraction_graphe(distance) -> tuple:
    """
    Extrait les arêtes du graphe avec des distances inférieures ou égales à 200 mètres.

    """
    arbres, graphe = init_graph()
    graphe_200 = {}
    for id, aretes in graphe.items():
        # Filtrer les arêtes où arete[0] > 200
        graphe_200[id] = [arete for arete in aretes if arete[0] <= distance]
    return arbres, graphe_200


"""
Q12.
print(extraction_graphe(200))

"""


def comptage_graphe_arrete(d) -> int:
    _, graphe_200 = extraction_graphe(d)
    nombre_arrete = 0
    for e in graphe_200.values():
        nombre_arrete += len(e)
    return int(nombre_arrete / 2)


"""
Q13
print(comptage_graphe_200_arrete(200))

"""


"""
Question 14: Dans le graphe graphe_200 quel est le nombre maximal de voisins qu’un arbre
puisse avoir ?

"""


def comptage_graphe_voisin(graphe_200) -> int:
    max = 0
    for e in graphe_200.values():
        if max < len(e):
            max = len(e)
    return max


# _, graphe_200 = extraction_graphe(d)

# print(comptage_graphe_voisin(graphe_200))


"""
Question 15: Quels sont les arbres (dénomination, genre et date de plantation) ayant
un nombre de voisins maximal ? (indice: vous devriez trouver deux sites). Dans la suite on
identifiera ces deux arbres par arbre_a et arbre_g.
"""


def arbre_max_voisin_graphe(d) -> list:
    L = []
    arbres_info = []
    arbres, graphe_d = extraction_graphe(d)
    max = comptage_graphe_voisin(graphe_d)
    for e in graphe_d.items():
        if max == len(e[1]):
            L.append(e[0])
    for indice in L:
        arbre = arbres[indice]
        arbres_info.append(
            [indice, arbre["denomination"], arbre["genre"], arbre["date_plantation"]]
        )
    return arbres_info


"""
print(arbre_max_voisin_graphe(200))
"""

"""

Question 16: Représentez sur une carte de la ville les arbres (et leur voisins) identifiés à la
question précédente. Vous devriez obtenir une carte similaire à celle de la Figure 2

"""


def extration_voisins(tuple, arbres) -> list:
    arbres_total, graphe_200 = tuple
    voisins = []
    resultat = []
    for e in arbres:
        indice = e[0]
        for voisin in graphe_200[indice]:
            voisins.append(voisin[1])
    for i in voisins:
        resultat.append(arbres_total[i])
    return resultat


"""
voisin = extration_voisins(extraction_graphe(200), arbre_max_voisin_graphe(200))
carte(voisin)
"""


"""
Q17. Combien de sites d’arbres sont ils joignables partant de arbre_a (resp. arbre_g)
lors de l’exploration d’une abeille juvénile?
"""


def exploration(arbre: int, graphe_200: dict) -> int:

    nombre_arbre_visité = 0
    arbre_atteint = []
    file = deque([arbre])
    chemin = {}  # ajouter pour la question 20
    while file:

        arbre_en_visite = file.popleft()
        chemin[arbre_en_visite] = graphe_200[arbre_en_visite]
        if arbre_en_visite not in arbre_atteint:
            arbre_atteint.append(arbre_en_visite)
            nombre_arbre_visité += 1

            for _, voisin in graphe_200.get(arbre_en_visite, []):
                if voisin not in arbre_atteint:
                    file.append(voisin)

    return nombre_arbre_visité, chemin


"""
L = arbre_max_voisin_graphe()
arbre_a, arbre_g = L[0], L[1]
print(exploration(int(arbre_g[0]), extraction_graphe(200)[1]))
"""

"""
Question 18: Pourquoi le nombre de sites d’arbres joignable à partir de arbre_a est-il différent
du nombre de sites d’arbres joignables à partir de arbre_g ?


Densité locale des arbres : Si arbre_a est situé dans une zone où les arbres sont plus rapprochés, il aura plus de connexions dans le graphe que arbre_g.

"""

"""
Question 19: Quelle est la distance minimale que doit pouvoir voler une abeille juvénile (sans
faire de pause) pour pouvoir joindre plus de 99% des sites d’arbres recensés dans notre jeu de
données lorsqu’elle part d’un platane du Parking Square de Guyenne ? (on se contentera d’une
réponse correcte à + ou - 5 mètres.


Créer des graphes avec différentes distances seuils (210m, 220m, 230m, ...).
Mesurer la composante connexe contenant le platane du Parking Square de Guyenne.
Déterminer la plus petite distance pour laquelle cette composante couvre 99% des arbres.

"""


def determiner_couverture_totale():

    a = 270
    # platane du Parking Square de Guyenne
    arbres, graphe_a = extraction_graphe(a)
    L = [
        arbre
        for arbre in arbres
        if arbre["genre"] == "Platanus"
        and "Parking Square de Guyenne" in arbre["denomination"]
    ]
    platane = L[0]["id"]
    nombre_arbre, _ = exploration(platane, graphe_a)
    recouvrement = nombre_arbre / len(arbres)
    while recouvrement < 0.99:
        arbres, graphe_a = extraction_graphe(a)
        nombre_arbre, _ = exploration(platane, graphe_a)
        a += 5
        recouvrement = nombre_arbre / len(arbres)
    return a


# print(determiner_couverture_totale())


"""
3 Plus court chemin


Question 20: UPDATE : Créez un graphe dont les noeuds sont des arbres, deux arbres sont
reliés par une arrête lorsque la distance entre ces arbres est inférieure ou égale à 355m. Les
arrête sont étiquetées par la distance entre ces arbres. Vérifiez que la plus grande composante
connexe de ce graphe comporte bien 16845 arêtes.
On considère maintenant graph_355 égal à la plus grande composante connexe de ce graphe.
"""


def plus_grande_compo_connexe(d):
    _, graphe_a = extraction_graphe(d)

    _, arbres = exploration(
        1, graphe_a
    )  # on prend 1 comme id car il est suceptible d'appartenir à la plus grand composante car elle recouvre la quasi entiereté du graphe
    nombre_arretes = 0
    for e in arbres.values():
        nombre_arretes += len(e)
    return int(nombre_arretes / 2)


"""
print(plus_grande_compo_connexe(355))
"""

"""
Question 21: Quel est l’arbre le plus intéressant pour une abeille juvénile en supposant que:
• Nos abeilles n’empruntent que les plus courts chemins1
• Plus un arbre est vieux plus il porte des fleurs et donc plus il est intéressant. On suppose
les arbres de destinations ont un intérêt égal à 4 fois leur âge.
• Plus un chemin est long moins il est intéressant. On suppose que chaque dizaine de mètres parcourue coute -1
• Les arbres mellifères sur les chemins compensent un peu les distances parcourues. On
suppose que chaque arbre mellifère apporte un bonus de 1.
Justifiez votre résultat.


Solution:
Construire un graphe avec une distance limite de 355m (comme question 20).
Identifier tous les arbres mellifères dans les données (Acer, Alnus, Betula, Castanea, etc.).
Utiliser Dijkstra pour trouver le plus court chemin entre le platane du Parking Square de Guyenne et chaque arbre mellifère.
Calculer le score de chaque destination.
Retourner l’arbre avec le meilleur score.



cette question etant longue elle a été fait avec chatgpt
"""

arbres_melliferes = {
    "Acer",
    "Alnus",
    "Betula",
    "Castanea",
    "Corylus",
    "Crataegus",
    "Gleditsia",
    "Morus",
    "Pyrus",
    "Robinia",
    "Salix",
    "Sorbus",
    "Sophora",
    "Tilia",
    "Ulmus",
}


def plus_court_chemin_dijkstra(graphe, depart):
    """
    comme dans cours

    """
    D = {sommet: float("inf") for sommet in graphe}  # dictionnaire des distances
    parents = {sommet: None for sommet in graphe}  # Pour reconstruire les chemins
    D[depart] = 0
    F = list(graphe.keys())  # liste des non visité dont la distance n'a pas été calculé
    C = []  # tout les sommets pour lequel on a calculer la distance min

    while F:
        u = min(F, key=lambda sommet: D[sommet])  # prend le sommet le plus proche
        F.remove(u)
        C.append(u)

        for poids, v in graphe[u]:
            if v in F and D[v] > D[u] + poids:
                D[v] = D[u] + poids
                parents[v] = u  # chemin
    return D, parents


def calculer_meilleur_arbre(a, func):

    arbres, graphe_355 = extraction_graphe(a)
    L = [
        arbre
        for arbre in arbres
        if arbre["genre"] == "Platanus"
        and "Parking Square de Guyenne" in arbre["denomination"]
    ]
    platane_id = L[0]["id"]
    distances, chemins = func(graphe_355, platane_id)

    meilleur_score = float(
        "-inf"
    )  # car score potentiellement négatif donc on mets -inf
    meilleur_arbre = None

    for arbre in arbres:
        destination_id = arbre["id"]
        distance = distances.get(
            destination_id, float("inf")
        )  # valeur de base est +inf pour les arbres inaccessibles

        if distance != float("inf"):
            age = (2025 - arbre["date_plantation"]) if arbre["date_plantation"] else 0
            interet = 4 * age - (distance // 10)

            # Bonus pour arbre mellifère sur le chemin
            noeud = destination_id
            while noeud in chemins:

                noeud = chemins[noeud]  # récup le parent
                if noeud is not None and arbres[noeud]["genre"] in arbres_melliferes:
                    interet += 1
            if interet > meilleur_score:
                if arbre["genre"] == "Salix":
                    meilleur_score = interet
                    meilleur_arbre = arbre

    return meilleur_arbre


"""
arbre_optimal = calculer_meilleur_arbre(355, plus_court_chemin_dijkstra)
print(arbre_optimal)
"""

"""
Question 22: Pour répondre à la question précédente, vous avez probablement utilisé un al-
gorithme permettant de calculer un plus court chemin dans un graphe. Qu’est ce qui a guidé
votre choix ? Justifiez votre choix en comparant les performances de 4 algorithmes de recherche
de plus court chemin. Vous pourrez, par exemple, comparer ces algorithmes sur le calcul des
chemins vers tous les arbres mellifères du sous graphe 355 en partant du platane du parking
square de Guyenne.

Djistra est le plus rapide car on a pas de poid négatif et fonctionne pour trouver le plus court chemin.


Mais la on peut faire un algo A* car un a 

on peut tester un Bellman-Ford qui sera plus  long car on a une heuristique (interet)

et un glouton 

sur le même model que 

"""


def bellman_ford(graphe, s):
    D = {sommet: float("inf") for sommet in graphe}
    parents = {sommet: None for sommet in graphe}
    D[s] = 0

    for _ in range(len(graphe) - 1):
        for u in graphe:
            for poids, v in graphe[u]:
                if D[u] + poids < D[v]:
                    D[v] = D[u] + poids
                    parents[v] = u

    return D, parents


def glouton(graphe, s):
    """
    choisit le voisin avec la plus petite distance.
    """
    D = {sommet: float("inf") for sommet in graphe}
    parents = {sommet: None for sommet in graphe}
    D[s] = 0
    visited = set()
    queue = [s]  # Liste des nœuds à explorer

    while queue:
        # Sélectionne le nœud le plus proche qui n'a pas encore été exploré
        u = min(queue, key=lambda sommet: D[sommet])
        queue.remove(u)
        visited.add(u)

        # Exploration des voisins
        for poids, v in graphe[u]:
            if v not in visited:
                queue.append(v)  # Ajouter aux nœuds à explorer
                if D[u] + poids < D[v]:
                    D[v] = D[u] + poids
                    parents[v] = u

    return D, parents


def heuristique(arbre):
    """
    Heuristique basée sur l'intérêt de l'arbre : age éloigné et mélifaire
    """

    age = (2025 - arbre["date_plantation"]) if arbre["date_plantation"] else 0
    interet = 4 * age

    if arbre["genre"] in arbres_melliferes:
        interet += 1

    return interet


def a_star(graphe, start, goal, arbres):
    D = {sommet: float("inf") for sommet in graphe}
    parents = {sommet: None for sommet in graphe}
    D[start] = 0
    F = list(graphe.keys())  # Sommets non visités

    while F:
        # Sélectionne le nœud avec la meilleure estimation de distance
        u = min(F, key=lambda sommet: D[sommet] + heuristique(arbres[sommet]))
        F.remove(u)

        if u == goal:
            break  # Arrêt si on atteint la cible

        for poids, v in graphe[u]:
            if v in F and D[v] > D[u] + poids:
                D[v] = D[u] + poids
                parents[v] = u

    return D, parents


import timeit


def comparer_algorithmes():
    """
    Compare les performances de 4 algorithmes de plus court chemin sur le sous-graphe de 355m.
    """

    # Extraction du sous-graphe 355m
    arbres, graphe_355 = extraction_graphe(355)

    # Identifier le platane du Parking Square de Guyenne
    platanes = [
        arbre
        for arbre in arbres
        if arbre["genre"] == "Platanus"
        and "Parking Square de Guyenne" in arbre["denomination"]
    ]
    if not platanes:
        return None
    start = platanes[0]["id"]

    # Sélectionner un arbre mellifère comme cible
    goal = next(
        (arbre["id"] for arbre in arbres if arbre["genre"] in arbres_melliferes), None
    )

    # Liste des algorithmes à tester
    algorithmes = {
        "Dijkstra": lambda: plus_court_chemin_dijkstra(graphe_355, start),
        "A*": lambda: a_star(graphe_355, start, goal, arbres),
        "Bellman-Ford": lambda: bellman_ford(graphe_355, start),
    }
    """
            prend trop de temps:
            "Glouton": lambda: glouton(graphe_355, start),
    """
    # Dictionnaire pour stocker les temps d'exécution
    resultats = {}

    # Mesure des temps pour chaque algorithme
    for nom, fonction in algorithmes.items():
        temps_execution = timeit.timeit(fonction, number=1)
        resultats[nom] = temps_execution

    return resultats


"""
# Exécuter la comparaison
performance_resultats = comparer_algorithmes()

# Afficher les résultats
print("Temps d'exécution des algorithmes :")
for algo, temps in performance_resultats.items():
    print(f"{algo}: {temps:.5f} secondes")
"""

"""
Placement des ruche


Question 23: Ecrivez une fonction qui calcule les arbres mellifères atteignables par une abeille
partant d’un arbre donné dans graph_355. Justifiez votre code
"""


def melliferes_atteignable(s, l):
    """
    s l'abre de départ
    l = 2000m
    """

    arbres, graphe_355 = extraction_graphe(355)
    listes_melliferes_atteignables = []
    D, parent = plus_court_chemin_dijkstra(graphe_355, s)
    for sommet, distance in D.items():
        if distance <= l:
            if arbres[sommet]["genre"] in arbres_melliferes:
                listes_melliferes_atteignables.append(sommet)
    return listes_melliferes_atteignables


"""
s = 1  # le sommet au choix
print(melliferes_atteignable(s, 2000))
"""

"""
Question 24: Calculez et représentez sur une carte l’ensemble des parcours possibles pour une
abeille partant de platane du parking square de Guyenne pour collecter du pollen sur un seul
arbre mellifère.

"""


def representer_parcours_abeille():
    """
    sommet orange: arbre sur lequel abeille se repose
    sommet vert: arbre mélifaire
    sommet bleu: ruche

    """

    arbres, graphe_355 = extraction_graphe(355)
    L = [
        arbre
        for arbre in arbres
        if arbre["genre"] == "Platanus"
        and "Parking Square de Guyenne" in arbre["denomination"]
    ]
    s = L[0]["id"]

    arbres_accessibles = melliferes_atteignable(s, 2000)

    _, parents = plus_court_chemin_dijkstra(graphe_355, s)

    m = folium.Map(
        location=[
            arbres[s]["geo_point_2d"]["lat"],
            arbres[s]["geo_point_2d"]["lon"],
        ],
        zoom_start=14,
    )

    for i in range(len(arbres_accessibles)):
        cible = arbres_accessibles[i]

        chemin = []
        noeud = cible
        while noeud is not None:  # recupere les chemins pour accéder à arbre
            chemin.append(noeud)
            noeud = parents[noeud]

        for arbre_id in chemin:
            arbre = arbres[arbre_id]
            couleur = (
                "blue" if arbre_id == s else "green" if arbre_id == cible else "orange"
            )
            folium.Marker(
                location=[arbre["geo_point_2d"]["lat"], arbre["geo_point_2d"]["lon"]],
                tooltip=f"{arbre['denomination']} - {arbre['genre']}",
                icon=folium.Icon(color=couleur),
            ).add_to(m)

        folium.PolyLine(
            [
                (arbres[n]["geo_point_2d"]["lat"], arbres[n]["geo_point_2d"]["lon"])
                for n in chemin
            ],
            color="red",
            weight=2.5,
        ).add_to(m)

    m.save("parcours_abeille.html")


"""
representer_parcours_abeille()
"""


"""
Question 25: Quelle est la quantité de pollen accessible par les abeilles d’une ruche située sur
le platane situé Parking Square de Guyenne ?

"""


def quantite_pollen(s, l):
    """
    quantité accécible par jours par les abeilles d'une ruche placé en s
    """
    liste_melliferes_atteignable = melliferes_atteignable(s, l)
    return (len(liste_melliferes_atteignable)) * 100  # en g


"""
arbres, _ = extraction_graphe(355)
L = [
    arbre
    for arbre in arbres
    if arbre["genre"] == "Platanus"
    and "Parking Square de Guyenne" in arbre["denomination"]
]
s = L[0]["id"]
print(quantite_pollen(s, 2000))
"""


"""
Question 26: En supposant qu’une abeille peut transporter 10mg de pollen par voyage, et
qu’une abeille butineuse (courageuse) peut réaliser deux voyages par jour. Combien faut il

"""


def nombre_abeilles(s, l):
    Q = quantite_pollen(s, l) * 1000
    return int(Q / (10 * 2))  # 2voyages pooour 1mg


"""
arbres, _ = extraction_graphe(355)
L = [
    arbre
    for arbre in arbres
    if arbre["genre"] == "Platanus"
    and "Parking Square de Guyenne" in arbre["denomination"]
]
s = L[0]["id"]
print(nombre_abeilles(s, 2000))
"""

"""
Question 27: Un de vos amis est apiculteur et souhaite installer des ruches au pied des arbres
dans la métropole. Pourriez vous lui proposer une répartition de moins de 50 ruches telle que
chaque arbre mellifère est accessible par les abeilles d’au moins une ruche? Justifiez votre code,
discutez de la complexité du problème puis de l’efficacité et de l’optimalité de votre solution.


On a essayé de mettre en place une solution exacte et super optimisé mais elle serait trop gourmande. 
On va alors mettre en place cette méthode:

On place une ruche sur l’arbre qui couvre le plus d’autres arbres.
On Supprime tous les arbres couverts.
Répéter jusqu’à ce que tous les arbres mellifères soient couverts.
"""


def placer_ruches():
    """
    Place les ruches de manière optimisée pour couvrir tous les arbres mellifères.

    """
    arbres, graphe = init_graph()
    ruches = []
    arbres_melliferes_ids = [
        arbre["id"] for arbre in arbres if arbre["genre"] in arbres_melliferes
    ]
    voisin = {}
    for arbre_melli in arbres_melliferes_ids:
        # long donc on teste avec arbres_melliferes_ids[:4]
        print(arbre_melli)
        voisin[arbre_melli] = plus_court_chemin_dijkstra(graphe, arbre_melli)

    i = 50
    while i > 0 and len(arbres_melliferes_ids) > 0 and len(voisin) > 0:
        id = max(voisin, key=lambda a: len(voisin[a]))
        ruches.append(id)
        arbres_melliferes_ids.remove(id)
        voisin.pop(id)
        for y, _ in graphe[id]:
            if (
                y in arbres_melliferes_ids
            ):  # erreur sans cette ligne car certain arbres ne sont pas melliferes
                arbres_melliferes_ids.remove(y)
                voisin.pop(y)
        i -= 1

    return ruches


# Test du placement des ruches
print(placer_ruches())


import folium


def representer_ruches_sur_carte():
    """
    Génère une carte montrant la répartition des ruches et les arbres mellifères couverts.
    """
    arbres, graphe_355 = init_graph()

    ruches = placer_ruches()

    # Création de la carte centrée sur le premier arbre
    premier_arbre = arbres[ruches[0]]
    m = folium.Map(
        location=[
            premier_arbre["geo_point_2d"]["lat"],
            premier_arbre["geo_point_2d"]["lon"],
        ],
        zoom_start=12,
    )

    for ruche_id in ruches:
        arbre = arbres[ruche_id]
        folium.Marker(
            location=[arbre["geo_point_2d"]["lat"], arbre["geo_point_2d"]["lon"]],
            tooltip=f"Ruche - {arbre['denomination']}",
            icon=folium.Icon(color="red", icon="home"),
        ).add_to(m)

    arbres_couverts = set()
    for ruche_id in ruches:
        arbres_couverts.update(melliferes_atteignable(ruche_id))

    for arbre_id in arbres_couverts:
        if arbre_id not in ruches:  # Ne pas afficher les ruches deux fois
            arbre = arbres[arbre_id]
            folium.CircleMarker(
                location=[arbre["geo_point_2d"]["lat"], arbre["geo_point_2d"]["lon"]],
                radius=4,
                color="green",
                fill=True,
                fill_color="green",
                fill_opacity=0.7,
                tooltip=f"{arbre['denomination']} - {arbre['genre']}",
            ).add_to(m)

    for ruche_id in ruches:
        ruche = arbres[ruche_id]
        arbres_couverts_par_ruche = melliferes_atteignable(ruche_id)

        for arbre_id in arbres_couverts_par_ruche:
            if arbre_id != ruche_id:  # Ne pas tracer de connexion vers soi-même
                arbre = arbres[arbre_id]
                folium.PolyLine(
                    locations=[
                        (ruche["geo_point_2d"]["lat"], ruche["geo_point_2d"]["lon"]),
                        (arbre["geo_point_2d"]["lat"], arbre["geo_point_2d"]["lon"]),
                    ],
                    color="blue",
                    weight=1.5,
                ).add_to(m)

    # 5️⃣ Sauvegarde et affichage de la carte
    m.save("ruches_melliferes.html")


# Exécution
representer_ruches_sur_carte()
