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
    Arbres = structuration()[1]
    for arbre1 in Arbres:
        for arbre2 in Arbres:
            if arbre1 != arbre2:
                D[
                    harversine(
                        arbre1["geo_point_2d"]["lat"],
                        arbre2["geo_point_2d"]["lat"],
                        arbre1["geo_point_2d"]["lon"],
                        arbre2["geo_point_2d"]["lon"],
                    )
                ] = (arbre1, arbre2)

    D = list((dict(sorted(D.items()))).items())
    d_max = D[-1]
    d_min = D[0]
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
        nombre_aretes_verif += len(e)
    return (
        nombre_noeuds == nombre_noeuds_verif,
        nombre_arretes == nombre_arretes_verif,
    )


"""
exercice 11:

print(verification(init_graph()))

"""


def extraction_graphe_200() -> tuple:
    """
    Extrait les arêtes du graphe avec des distances inférieures ou égales à 200 mètres.

    Returns:
        Dict[int, List[Tuple[float, int]]]: Graphe filtré.
    """
    arbre, graphe = init_graph()
    graphe_200 = {}
    for id, aretes in graphe.items():
        # Filtrer les arêtes où arete[0] > 200
        graphe_200[id] = [arete for arete in aretes if arete[0] <= 200]
    return arbre, graphe_200


"""
Q12.
print(extraction_graphe_200())

"""


def comptage_graphe_200_arrete() -> int:
    _, graphe_200 = extraction_graphe_200()
    nombre_arrete = 0
    for e in graphe_200.values():
        nombre_arrete += len(e)
    return int(nombre_arrete / 2)


"""
Q13
print(comptage_graphe_200_arrete())

"""


"""
Question 14: Dans le graphe graphe_200 quel est le nombre maximal de voisins qu’un arbre
puisse avoir ?

"""


def comptage_graphe_200_voisin() -> int:
    max = 0
    _, graphe_200 = extraction_graphe_200()
    for e in graphe_200.values():
        if max < len(e):
            max = len(e)
    return max


# print(comptage_graphe_200())


"""
Question 15: Quels sont les arbres (dénomination, genre et date de plantation) ayant
un nombre de voisins maximal ? (indice: vous devriez trouver deux sites). Dans la suite on
identifiera ces deux arbres par arbre_a et arbre_g.
"""


def arbre_max_voisin_graphe_200() -> list:
    L = []
    arbres_info = []
    max = 0
    arbres, graphe_200 = extraction_graphe_200()
    for e in graphe_200.items():
        if max < len(e[1]):
            max = len(e[1])
            L = [e[0]]
        elif max == len(e[1]):
            L.append(e[0])
    for indice in L:
        arbre = arbres[indice]
        arbres_info.append(
            [indice, arbre["denomination"], arbre["genre"], arbre["date_plantation"]]
        )
    return arbres_info


"""
print(comptage_graphe_200())
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
voisin = extration_voisins(extraction_graphe_200(), comptage_graphe_200())
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

    while file:
        arbre_en_visite = file.popleft()
        if arbre_en_visite not in arbre_atteint:
            arbre_atteint.append(arbre_en_visite)
            nombre_arbre_visité += 1

            for _, voisin in graphe_200.get(arbre_en_visite, []):
                if voisin not in arbre_atteint:
                    file.append(voisin)

    return nombre_arbre_visité


L = arbre_max_voisin_graphe_200()
arbre_a, arbre_g = L[0], L[1]
print(exploration(int(arbre_a[0]), extraction_graphe_200()[1]))


"""
Question 18: Pourquoi le nombre de sites d’arbres joignable à partir de arbre_a est-il différent
du nombre de sites d’arbres joignables à partir de arbre_g ?

"""

"""
Question 19: Quelle est la distance minimale que doit pouvoir voler une abeille juvénile (sans
faire de pause) pour pouvoir joindre plus de 99% des sites d’arbres recensés dans notre jeu de
données lorsqu’elle part d’un platane du Parking Square de Guyenne ? (on se contentera d’une
réponse correcte à + ou - 5 mètres.

"""
