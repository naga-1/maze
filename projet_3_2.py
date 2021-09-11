from random import *             # utilisé pour randint
from numpy import *              # utilisé pour manipuler plus facilement les arrays
from time import *               # utilisé pour calculer le temps de génération
import matplotlib.pyplot as plt  # utilisé pour afficher une image à partie d'un tableau

# 0 = mur   1 = vide
############################################# partie 1 #################################################################


def display_maze(maze, title):
    """ affiche le labyrinthe avec un title sous forme d'image, 0=noir, 1=blanc, 2=rouge, 3=bleu, 4=vert
        sert pour afficher les murs en noir, les chemins en blanc, le but en rouge, le point de depart en bleu et
        le chemin le plus court en vert"""

    # création d'un tableau de tableaux nulles de taille le labyrinthe
    tableau = [[[0] for i in range(len(maze))] for j in range(len(maze))]

    for i in range(len(maze)):    # on parcours le labyrinthe
        for j in range(len(maze)):
            # si la case vaut 1, on rentre la couleur blanche à cette position dans le tableau
            if maze[i][j] == 1:
                tableau[i][j] = [255, 255, 255]
            # si la case vaut 0, on rentre la couleur noire à cette position dans le tableau
            elif maze[i][j] == 0:
                tableau[i][j] = [0, 0, 0]
            # si la case vaut 4, on rentre la couleur verte à cette position dans le tableau
            elif maze[i][j] == 4:
                tableau[i][j] = [0, 255, 0]
            # si la case vaut 3, on rentre la couleur bleu à cette position dans le tableau
            elif maze[i][j] == 3:
                tableau[i][j] = [0, 0, 255]
            # si la case vaut 5, on rentre la couleur correspondant à une phéromone
            elif maze[i][j] == 5:
                tableau[i][j] = [125, 125, 125]
            # sinon on rentre la couleur rouge à cette position dans le tableau
            else:
                tableau[i][j] = [255, 0, 0]



    plt.imshow(tableau)     # on ajoute l'image
    plt.title(title)        # on ajoute le titre
    return plt.show()       # on affiche l'image avec le titre


def initialize_maze(taille):
    """ initialise le tableau avec plein de zeros (murs) et le return"""
    tableau = zeros((taille, taille), int)

    return tableau


def add_goal_maze(maze):
    """ajoute une cible rouge à atteindre dans le labyrinthe et return le labyrinthe modifié"""

    pos_i = randint(0, len(maze) - 1)           # choix d'une ligne au hasard
    pos_j = randint(0, len(maze[0]) - 1)        # choix d'une colonne au hasard

    while maze[pos_i][pos_j] != 1:              # tant que on tombe pas sur une case vide
        pos_i = randint(0, len(maze) - 1)       # choix d'uneligne au hasard
        pos_j = randint(0, len(maze[0]) - 1)    # choix d'unecolonne au hasard

    maze[pos_i][pos_j] = 2  # modification de la case choisie par la cible dans le labyrinthe

    return maze


def voisins_valides(maze, i, j):
    """return la liste des voisins valides de i,j, c'est à dire les voisins égales à 0"""

    liste_voisin = []

    for ligne in range(3):  # parcourt i,j et le carré de 3 * 3 autour de i,j
        for colonne in range(3):
            # si les coordonnées ne valent pas (i,j)
            if not (i + 1 - ligne == i and j + 1 - colonne == j):
                # si les coordonnées sont dans le tableau
                if 0 <= (i + 1 - ligne) < len(maze) and 0 <= (j + 1 - colonne) < len(maze):
                    # si la case correspondante à un 0
                    if maze[i + 1 - ligne][j + 1 - colonne] == 0:
                        # on ajoute ces cordonnées dans la lite des voisins valides
                        liste_voisin.append([i + 1 - ligne, j + 1 - colonne])

    return liste_voisin


def voisin_eligible_bool(maze, i, j):
    """ sous fonction de voisins_eligibles
    return True si parmis les voisins de i,j il y a 1 ou 0 voisins qui ont la valeur 1, False sinon"""
    counter = 0
    for ligne in range(3):  # parcourt i,j et le carré de 3 * 3 autour de i,j
        for colonne in range(3):
            # si les coordonnées sont dans le tableau
            if 0 <= (i + 1 - ligne) < len(maze) and 0 <= (j + 1 - colonne) < len(maze):
                # si la case correpondante à ces coordonnées vaut 1
                if maze[i + 1 - ligne][j + 1 - colonne] == 1:
                    # on incrémente le compteur
                    counter += 1

    if counter > 1:  # si le compteur > 1, il y a plus de 1 voisin déja à 1, donc voisin pas eligible
        return False
    else:
        return True


def voisins_eligibles(maze, liste_voisin_valide):
    """ prend la liste des voisins valide et return la liste des voisins éligibles"""

    liste_voisins_eligible = []

    for voisin in liste_voisin_valide:  # on parcours les voisins parmi les voisins valides
        # si ces voisins sont éligibles
        if voisin_eligible_bool(maze, voisin[0], voisin[1]):
            # on les rajoute dans la liste des voisins éligibles
            liste_voisins_eligible.append(voisin)

    return liste_voisins_eligible


def generate_maze(taille):
    """ fonction qui génère aléatoirement et de manière itérative un labyrinthe
        s'appuie sur les sous fonctions voisins_valides et voisins_éligibles, prend en entrée un labyrinthe vierge
        et return un labyrinthe créé"""

    maze = initialize_maze(taille)

    liste_voisins_valides = []          # init pour la comprehension
    liste_voisins_eligibles = []        # init pour la comprehension
    choix = 0                           # init, variable pour choisir un voisin aléatoirement
    pile = []                           # init pile

    pos_i = randint(0, len(maze)-1)        # choix d'une ligne au hasard
    pos_j = randint(0, len(maze[0])-1)     # choix d'une colonne au hasard

    maze[pos_i][pos_j] = 1        # modification de la case choisie dans le labyrinthe
    pile.append((pos_i, pos_j))  # ajoute la case dans la pile

    while pile:     # tant qu'on a des couples de coordonnées dans la pile

        liste_voisins_valides = voisins_valides(maze, pos_i, pos_j)
        liste_voisins_eligibles = voisins_eligibles(maze, liste_voisins_valides)

        if liste_voisins_eligibles:    # s'il y'a des couples de coordonnées dans la liste des voisins eligibles

            # on choisit un couple au hasard dans la liste

            # utilise randint de random
            choix = randint(0, len(liste_voisins_eligibles)-1)  # on choisit un des couples au hasards
            pos_i = liste_voisins_eligibles[choix][0]
            pos_j = liste_voisins_eligibles[choix][1]

            maze[pos_i][pos_j] = 1          # modification dans le labyrinthe de la case correspondante au couple choisi
            pile.append((pos_i, pos_j))    # ajout du couple de coordonnées dans la pile

        else:  # si pas de voisins éligibles
            pile.remove((pos_i, pos_j))  # on enleve le couple de coordonne qu'on étudiait
            if pile:                     # si il reste des éléments dans la pile
                pos_i = pile[-1][0]        # on prend le couple de coordonnées juste avant dans la pile, cad le dernier
                pos_j = pile[-1][1]

    return maze   # on retourne le labyrinthe


def display_colored_map(maze):
    """ affiche et enregistre la map colorée à partir de la map réalisée avec drijkstra"""
    fig, ax = plt.subplots()
    ax.matshow(maze)
    plt.title("maze colored")
    plt.savefig("maze_colored.png")
    plt.show()


def add_start_point_maze(maze):
    """ajoute une cas de départ bleu à atteindre dans le labyrinthe et return le labyrinthe modifié"""

    pos_i = randint(0, len(maze) - 1)           # choix d'une ligne au hasard
    pos_j = randint(0, len(maze[0]) - 1)        # choix d'une colonne au hasard

    while maze[pos_i][pos_j] != 1:              # tant que on tombe pas sur une case vide
        pos_i = randint(0, len(maze) - 1)       # choix d'une ligne au hasard
        pos_j = randint(0, len(maze[0]) - 1)    # choix d'une colonne au hasard

    maze[pos_i][pos_j] = 3  # modification de la case choisie par le point de depart dans le labyrinthe

    return maze


############################################# partie 2 #################################################################


def dijkstra(maze):
    """ Prend en entrée un labyrinthe et sort la map directionnelle associée, utilise l'algorithme de dijkstra fourni"""

    # création et initialisation de la map directionelle représenant le labyrinthe à None
    map_dijkstra = [[[None] for i in range(len(maze))] for j in range(len(maze))]

    nb_value = 0                        # initialisation du compteur qui va compter le nombre de cases qu'on a modifié
    nb_max = len(maze)*len(maze)        # nombre de cases dans le labyrinthe

    # parcours le labyrinthe et associe dans la map les murs à la valeur -1 et le but à la valeur 0
    for a in range(len(maze)):
        for b in range(len(maze)):
            if maze[a][b] == 0:     # mur
                map_dijkstra[a][b] = -1
                nb_value += 1
            elif maze[a][b] == 2:   # goal_point
                map_dijkstra[a][b] = 0
                nb_value += 1

    counter = 0        # compteur à 0

    # tant qu'on a pas modifié toutes les cases, c'est à dire s'il reste des cases à None,on continue
    while nb_value != nb_max:
        # parcours la map et cherche s'il y'a des cases égalent à "counter" pour assigner la valeur "counter-1"
        # aux cases voisines si elles n'ont pas déjà de valeur, puis décrémente "counter" de 1
        for ligne in range(len(map_dijkstra)):
            for colonne in range(len(map_dijkstra)):
                if map_dijkstra[ligne][colonne] == counter:
                    # parcourt i,j et le carré de 3 * 3 autour de i,j
                    for i in range(3):
                        for j in range(3):
                            # si les coordonnées sont dans la map
                            if 0 <= (ligne + 1 - i) < len(map_dijkstra) and 0 <= (colonne + 1 - j) < len(map_dijkstra):
                                # si la valeur correspondante n'a pas de valeur (alors c'est un None)
                                # if map_dijkstra[ligne + 1 - i][colonne + 1 - j] is None: ne marche pas
                                if not isinstance(map_dijkstra[ligne + 1 - i][colonne + 1 - j], int):

                                    map_dijkstra[ligne + 1 - i][colonne + 1 - j] = counter+1
                                    nb_value += 1

        counter += 1  # décrémentation du compteur

    # display(map_dijkstra)
    # print(map_dijkstra)
    # complexité environ = à x carré + 18 fois x puissance 4
    return map_dijkstra


def dijkstra_to_mapdir(map_dijkstra):
    """ prend en entrée la map sortie par dijkstra et sort la map directionnelle associée
        (la case pointe sur une case adjacente où se déplacer pour rejoindre le but)"""

    # on initialise la map directionnelle
    map_dir = [[[0] for i in range(len(map_dijkstra))] for j in range(len(map_dijkstra))]

    # on copie la map dijkstra, comme ça on a les murs
    for ligne in range(len(map_dijkstra)):
        for colonne in range(len(map_dijkstra)):
            map_dir[ligne][colonne] = map_dijkstra[ligne][colonne]

    # on parcours la map directionnelle
    for ligne in range(len(map_dir)):
        for colonne in range (len(map_dir)):
            # si on est sur une case (pas un mur) on initialise les variables pour trouver le minimum parmi les voisines
            if map_dijkstra[ligne][colonne] != -1 :
                mini_coord = (0, 0)
                mini = inf
                # on parcourt la case (ligne, colonne) et ses voisins pour trouver la case de valeur minimum
                for i in range(3):
                    for j in range(3):
                        x = ligne + 1 - i
                        y = colonne + 1 - j
                        # si les coordonnées sont dans la map
                        if 0 <= x < len(map_dir) and 0 <= y < len(map_dir):
                            # si la coordonnée n'est pas un mur et à un numéro inférieur à ceux parcourus
                            if map_dijkstra[x][y] != -1 and map_dijkstra[x][y] < mini:
                                mini = map_dijkstra[x][y]
                                mini_coord = (x, y)
                # on marque dans la map directionnelle les coordonnées de la case qui à le numéro le plus faible
                map_dir[ligne][colonne]=mini_coord

    # on retourne la map directionnelle
    return map_dir


def resol_maze_mapdir(maze):
    """fonction qui prend en entrée un labyrinthe avec un but et un départ, le résoud avec la carte directionnel
        return le labyrinthe avec le chemin le plus court entre les 2 points"""

    # on calcule la map directinnelle
    map_dir = dijkstra_to_mapdir(dijkstra(maze))

    start_find = False  # valeur booléenne à False tant qu'on a pas trouvé la case de départ
    i_start = 0
    j_start = 0

    # recherche de la case de départ, on parcourt le labyrinthe
    for ligne in range(len(maze)):
        for colonne in range(len(maze)):
            # si la valeur de la case vaut 3, c'est à dire c'est la case de départ
            if maze[ligne][colonne] == 3:
                i_start = ligne  # on copie la valeur i de la case
                j_start = colonne  # on copie la valeur j de la case
                start_find = True  # on passe à True pour sortir du double for
                break
        if start_find:
            break

    # avec la map directionnelle on va sur la case suivante pointée par la case où on se trouve
    i_suivant, j_suivant = map_dir[i_start][j_start]

    # tant qu la case suivante dans le labyrinthe n'est pas le goal
    while maze[i_suivant][j_suivant] != 2:

        # on associe la valeur 4 dans le labyrinthe (pour pouvoir afficher le chemin)
        maze[i_suivant][j_suivant] = 4
        # avec la map directionnelle on va sur la case suivante pointée par la case où on se trouve
        i_suivant, j_suivant = map_dir[i_suivant][j_suivant]

    # on retourne le labyrinthe modifié
    return maze


def resol_maze_dijkstra(maze):
    """ fonction qui prend en entrée un labyrinthe avec un but et un départ, le résoud avce dijkstra
        return le labyrinthe avec le chemin le plus court entre les 2 points"""

    map_dir = dijkstra(maze)      # on veut la map valuée en fonction de la case à atteindre (but) donné par dijkstra
    start_find = False            # valeur booléenne à False tant qu'on a pas trouvé la case de départ
    i_start = 0
    j_start = 0



    # recherche de la case de départ, on parcourt le labyrinthe
    for ligne in range(len(maze)):
        for colonne in range(len(maze)):
            # si la valeur de la case vaut 3, c'est à dire c'est la case de départ
            if maze[ligne][colonne] == 3:
                i_start = ligne             # on copie la valeur i de la case
                j_start = colonne           # on copie la valeur j de la case
                start_find = True           # on passe à True pour sortir du double for
                break

        if start_find:
            break

    longueur = map_dir[i_start][j_start]
    counter = map_dir[i_start][j_start]  # le compteur commence à la valeur correspondant à la case de départ dans map

    i_suivant = i_start
    j_suivant = j_start

    # tant que le compteur ne vaut pas 1 (c'est à dire qu'on est à une case de l'arrivée)
    while counter != 1:

        # parcourt la case (i_suivant,j_suivant) et le carré de 3 * 3 autour d'elle
        for ligne in range(3):
            for colonne in range(3):
                i = i_suivant + 1 - ligne
                j = j_suivant + 1 - colonne
                # si les coordonnées sont dans le tableau
                if 0 <= i < len(maze) and 0 <= j < len(maze):
                    # si la valeur correspondante dans map est counter-1 (c'est à dire se rapproche du but)
                    if map_dir[i][j] == counter-1:
                        i_suivant = i          # on prend ces coordonnée pour aller dessus
                        j_suivant = j
                        maze[i][j] = 4         # on modifie la valeur de cette case par 4 dans le labyrinthe
                        break

        counter -= 1    # on décremente de 1 le counter

    return maze, longueur


############################################# partie 3 #################################################################


def add_goal_and_start_points_for_algo_gene(maze):
    """Prend un labyrinthe sans but et sans point de départ.
        Va choisir un but et un départ assez éloignés, puis renvoyer le labyrinthe modifié"""

    # ajout d'un but au labyrinthe
    maze = add_goal_maze(maze)

    # calcul de la map dijkstra (distance au but) du labyrinthe
    map_dijkstra = dijkstra(maze.copy())

    # on choisit un chemin de longeur minimum entre le départ et le but
    mini = len(maze) * len(maze) * 0.06

    # on regarde si cette longueur est possible dans le labyrinthe créé
    ok = False
    for ligne in map_dijkstra:
        for elem in ligne:
            if elem >= mini:
                ok = True
                break
        if ok:
            break
    if not ok:
        print("erreur, veuillez relancer un autre labyrinthe")
        return -1

    # ajoute un point de depart pas trop près
    pos_i = randint(0, len(maze) - 1)  # choix d'une ligne au hasard
    pos_j = randint(0, len(maze[0]) - 1)  # choix d'une colonne au hasard

    # tant que le point de départ n'est pas assez éloigné du but, on en prend un autre
    while map_dijkstra[pos_i][pos_j] < mini:
        pos_i = randint(0, len(maze) - 1)  # choix d'une ligne au hasard
        pos_j = randint(0, len(maze[0]) - 1)  # choix d'une colonne au hasard

    # on note le point de depart dans le labyrinthe
    maze[pos_i][pos_j] = 3

    return maze


def generation(maze, len_mini):
    """ génère une centaine de programmes/chemins avec des directions aléatoires
        Prend en entrée un labyrinthe avec départ et but ainsi que la longueur minimale des chemins.
        Puis va assigner une longueur ( >= distance départ - but) au programmes/chemins et leurs assigner des
        directions aléatoires (de 0 à 7). Enfin on retourne un tableau contenant tous ces programmes/chemins"""

    # on prend au hasard le nombre de programmes/chemins et leur longueur
    nbr_p = randint(100, 120)
    len_p = randint(len_mini +1, len(maze)**2)

    # on initialise tous les programmes/chemins dans un tableau
    tableau_p = [[randint(0, 7) for i in range(len_p + 1)] for j in range(nbr_p)]

    #print(tableau_p)
    return tableau_p  #, coord,maze


def suivant(dir, i, j):
    """ prend un direction (de 0 à 7) et une position (i,j) et ressort la position correspondante à la case suivante.
        La case suivante est devinée en fonction de la position de la case précédente et de la direction donnée
        sous fonction pour end_cell"""
    if dir == 0:
        return i, j+1
    elif dir == 1:
        return i-1, j+1
    elif dir == 2:
        return i-1, j
    elif dir == 3:
        return i-1, j-1
    elif dir == 4:
        return i, j-1
    elif dir == 5:
        return i+1, j-1
    elif dir == 6:
        return i+1, j
    else:
        return i+1, j+1


def one_neighbour(maze, pos_i, pos_j):
    """ regarde si la case sur laquelle on est a un voisin seulement -> si on est dans une impasse
        Reçoit le labyrinthe, les coordonnées de la case et renvoie true ou false
        sous fonction pour pose_pheromone"""

    cod_i = pos_i
    cod_j = pos_j

    counter = 0
    for ligne in range(3):  # parcourt i,j et le carré de 3 * 3 autour de i,j
        for colonne in range(3):
            if not (pos_i + 1 - ligne == cod_i and pos_j + 1 - colonne == cod_j):
                # si les coordonnées sont dans le tableau
                if 0 <= (pos_i + 1 - ligne) < len(maze) and 0 <= (pos_j + 1 - colonne) < len(maze):
                    # si la case correpondante à ces coordonnées vaut 1,2 ou 3 (chemin libre, case arrivée, case depart)
                    if maze[pos_i + 1 - ligne][pos_j + 1 - colonne] in (1,2,3):
                        # on incrémente le compteur
                        counter += 1

    if counter == 1:  # si le compteur = 1, la cellule n'a que un chemin où aller (cul de sac)
        return True
    else:
        return False


def pose_pheromone(laby, memory, pos_i, pos_j, pos_i_d, pos_j_d):
    """ si le chemin C est dans une impasse, on pose des pheromones pour eviter que les autres chemins passent par là
        Priend en entrée le labyrinthe, la mémoire des cases parcouruent par le chemin, les positions des cases
        d'arivées et de départ.
        sous fonction dans end_cell"""

    maze = laby.copy()

    # si la case n'a que 1 voisin et n'est pas la case depart (il ne faut pas colorier la case de depart)
    if one_neighbour(maze.copy(), pos_i, pos_j) and (pos_i_d, pos_j_d) != (pos_i, pos_j):
        # tant que la case est dans une impasse et qu'elle n'est pas sur les cases d'arrivée ou de depart
        while one_neighbour(maze.copy(), pos_i, pos_j) and maze[pos_i][pos_j] not in (2,3):
            # alors on continue à poser une phéromone et on passe à la case d'après
            maze[pos_i][pos_j] = 5
            memory.pop()
            pos_i, pos_j = memory[-1]
            # tant que la nouvelle case est déja colorée (si retour en arrière) on prend la suivante
            while maze[pos_i][pos_j] == 5:
                memory.pop()
                pos_i, pos_j = memory[-1]

    return pos_i, pos_j, maze


def end_cell(C, maze, coord_depart):
    """ prend un programme/chemin C en entrée, le labyrinthe, les coordonnées de la case de départ.
        Si le programme est arrivé dans un cul de sac, on modifie le labyrinthe en posant des pheromones
        Sort la position de la dernière case valide du chemin et le labyrinthe modifié"""

    # on recupere et on assigne les coordonnées de la position de depart
    pos_i = coord_depart[0]
    pos_j = coord_depart[1]
    # on copie toutes les cases que l'on parcourt, on l'utilise dans pose pheromone
    memory = [(pos_i,pos_j)]

    # on test les cases pointées par les instructions du programme/chemin C, tant que la case est valide on continue
    for direction in C:

        # on va prendre les coordonnées de la prochaine case et les tester
        pos_i_test, pos_j_test = suivant(direction, pos_i, pos_j)

        # si les coordonnées sont dans le labyrinthe et ne pointent pas sur un mur ou un chemin barré/interdit
        if 0 <= pos_i_test < len(maze) and 0 <= pos_j_test < len(maze) and maze[pos_i_test][pos_j_test] != 0 and maze[pos_i_test][pos_j_test] != 5:
            # si les coordonnées sont égales au goal on retourne cette position
            if maze[pos_i_test][pos_j_test] == 2:
                #print("got it",pos_i_test, pos_j_test)
                return pos_i_test, pos_j_test, maze
            # sinon on enregistre cette position et on passe à la position d'après vu que celle là est valide
            else:
                pos_i, pos_j = pos_i_test, pos_j_test
                memory.append((pos_i,pos_j))
        # sinon on regarde si on est arrivé dans une impasse, on utilise pose_pheronome
        # qui renvoit le nouveau labyrinthe et la dernière case valide après modification du labybrinthe
        else:
            pos_i, pos_j, maze = pose_pheromone(maze.copy(), memory, pos_i, pos_j, coord_depart[0],coord_depart[1])
            return pos_i, pos_j, maze

    pos_i, pos_j, maze = pose_pheromone(maze.copy(), memory, pos_i, pos_j, coord_depart[0], coord_depart[1])
    return pos_i, pos_j, maze


def dist(coord_c, coord_g):
    """ prend en entrée les coordonnées d'une case et de la case à atteindre
        renvoie la distance entre ces deux case sqrt( (x_1 - x_2)**2 + (y_1 - y_2)**2 )"""
    return sqrt((coord_c[0] - coord_g[0])**2 + (coord_c[1] - coord_g[1])**2)


def penalties(C, maze, coord_depart,coord_goal):
    """ Prend en entrée un chemin C, le labyrinthe et les coordonnées de départ et d'arrivée
        calcul le nombre de cases identiques visitées et ressort une pénalité associée
        (longueur de c moins le nbr de cases différentes parcourue) fois la distance (départ-arrivée)
        -> favorise les longs chemins, peut etre un pb dans certains cas"""

    # distance est la distance entre les cases départ et arrivées, servira de poids pour calculer les pénalités
    distance = dist(coord_depart,coord_goal)
    #distance = len(C)-dist(coord_depart,coord_goal)
    nbr_identique = 0                           # nombre de cellule identiques visitées
    nbr_total = 1                               # nombre de cellues totales visitées

    pos_i = coord_depart[0]
    pos_j = coord_depart[1]

    parcourue = [(pos_i, pos_j)]                # tableau référençant les cellules visitées

    # test les cases pointées par les instructions du programme/chemin C, tant que la case est valide on continue
    for direction in C:

        # on va prendre les coordonnées de la prochaine case et les tester
        pos_i_test, pos_j_test = suivant(direction, pos_i, pos_j)

        # si les coordonnées sont dans le labyrinthe et ne pointent pas sur un mur ou une phéromone
        if 0 <= pos_i_test < len(maze) and 0 <= pos_j_test < len(maze) and maze[pos_i_test][pos_j_test] != 0 and maze[pos_i_test][pos_j_test] != 5:
            # si les coordonnées sont égales au goal
            if maze[pos_i_test][pos_j_test] == 2:
                nbr_total += 1
                return (len(C)-nbr_total+nbr_identique) * distance
            # sinon on enregistre cette position et on passe à la position d'après vu que celle là est valide
            else:
                # si la cellule est déja visité on incrémente le compteur
                if (pos_i_test, pos_j_test) in parcourue:
                    nbr_identique += 1
                # sinon on l'ajoute à cette liste
                else:
                    parcourue.append((pos_i_test, pos_j_test))
                nbr_total += 1
                pos_i, pos_j = pos_i_test, pos_j_test
        # sinon on retourne le résulat
        else:
            return (len(C)-nbr_total+nbr_identique) * distance

    return (len(C)-nbr_total+nbr_identique) * distance


def fitness(C,  maze, coord_goal, coord_depart):
    """ fonction qui calcule la fitness d'un chemin, prend en entrée un chemin, le labyrinthe, les coordonnées des cases
        de départs, retourne un nombre >= 0. S'appuie sur les fonctions en_cell, dist et penalties.
        plus le résultat est éloigné de 0 plus le chemin est mauvais"""

    # end_cell retourne la dernière position valide dans le chemin c et le labyrinthe modifié avec les phéromones
    pos_i_c, pos_j_c, maze = end_cell(C, maze.copy(), coord_depart)

    return dist((pos_i_c, pos_j_c), coord_goal) + penalties(C, maze, coord_depart,coord_goal), maze.copy()
    # distance entre l'arrivée et la dernière case plus des pénalitées


def selection(tableau_p, ts, maze):
    """ selectionne les chemins entrés grâce à la fonction fitness, ressort les chemins/programmes selectionnés
        Prend en entrée le tableau des chemins, le taux de selection et le labyrinthe"""

    # trouve les coordonnées des cases de départ et d'arrivée dans le labyrinthe
    for i in range(len(maze)):
        for j in range(len(maze)):
            if maze[i][j] == 2:
                coord_goal = (i,j)
            elif maze[i][j] == 3:
                coord_d = (i,j)

    valeur_moy = 0                      # valeur moyenne de la fitness pour une génération
    tableau_p_value =[]                 # tableau qui va stocker les valeurs de fitness des chemins (dans l'ordre)

    # on calcule la valeur de fitness de chaque chemin et on la stocke, on calcule aussi la moyenne
    for chemin in tableau_p:
        value, maze = fitness(chemin,  maze.copy(), coord_goal, coord_d)
        tableau_p_value.append(value)

        valeur_moy += value

    valeur_moy = valeur_moy/len(tableau_p)          # moyenne de la valuer de fitness

    # on trie le tableau de chemins en s'aidant des valeurs de fitness avec un tri par insertion
    for i in range(0, len(tableau_p_value)):                    # on parcourt tout le tableau des valeurs corespondantes au chemins
        for j in range(0, i):                                   # on compare parmi les éléments du tableau de valeur déjà triée
            if tableau_p_value[i] < tableau_p_value[j]:         # si la valeur de cet élément est inférieure à la valeur d'un autre élément
                tableau_p_value.insert(j, tableau_p_value[i])   # on insert la valeur de cet élément juste avant la valeur de l'autre élément
                tableau_p.insert(j, tableau_p[i])               # on insert le chemin équivalent à cet élément juste avant le chemin équivalent de l'autre élément
                del tableau_p_value[i + 1]                      # puis on supprime l'ancienne position de l'élément qu'on vient d'insérer
                del tableau_p[i + 1]                            # puis on supprime l'ancienne position du chemin qu'on vient d'insérer
                break

    # on calcule avec le taux de selection le nombre de chemins à conserver
    taux_selection = int(len(tableau_p)*ts)
    Nombre_enfant = int(len(tableau_p) - taux_selection)
    # on retourne le tableau des chemins conservés et le nombre d'enfants à créer
    return tableau_p[:taux_selection], Nombre_enfant, valeur_moy, maze


def reproduction(tableau_p, Ne):
    """ fonction de reproduction qui prend le tableau des chemins sélectionnés avec fitness et le nombre d'enfants,
        ressort le tableau avec des nouveaux individus"""

    len_choice = len(tableau_p) - 1

    # on selectionne des parents au hasard , on prend environ la moitié de chaques parents et on créer un enfant
    for i in range(Ne):
        parent_1 = tableau_p[randint(0 ,len_choice)]
        parent_2 = tableau_p[randint(0, len_choice)]
        cut = randint(int(len(parent_1)/2 - len(parent_1)/10), int(len(parent_1)/2 + len(parent_1)/10))
        enfant = parent_1[:cut] + parent_2[cut:]
        tableau_p.append(enfant)

    # on retourne le tableau avec les parents et les enfants
    return tableau_p


def mutation(tableau_p, tm):
    """ prend en entrée le tableau des chemins et un taux de mutation, ressort le tableau de chemin avec des mutations
        dans les chemins, en fonction du taux de mutation"""

    taux_mutation = int(len(tableau_p) * tm)

    # on modifie au hasard un gene à une position au hasard , dans un chemin au hasard. autant de fois que le tm fournit
    for i in range(taux_mutation):
        chemin = randint(0, len(tableau_p)-1)
        position = randint(0, len(tableau_p[chemin])-1)
        direction = randint(0,7)
        tableau_p[chemin][position] = direction

    return tableau_p


def resol_maze_genetic(maze, ts, tm):
    """ resoud un labyrinthe avec un algorithme génétique, prend un labyrinthe sans but et sans goal et
        ressort le labyrinthe avec le chemin. S'appuie sur différentes sous fonctions"""

    # trouve les coord de depart
    find = False
    for i in range(len(maze)):
        for j in range(len(maze)):
            if maze[i][j] == 3:
                coord_depart = (i, j)
                find = True
                break
        if find:
            break

    #print(maze.copy()) # affiche le labyrinthe dans le terminal

    # trouve la longueur minimum pour les chemins
    len_mini = dijkstra(maze.copy())[coord_depart[0]][coord_depart[1]]
    #len_mini = len(maze)**2
    Nbr_gene = 0            # calcule le nombre de generations

    # generation des chemins
    tableau_p = generation(maze.copy(), len_mini)

    value_moy = []  # tableau qui prend les valeurs de fitness moyennes (pour tracer une courbe)
    gen = []        # tableau qui prend le nombre de generations (pour tracer une courbe)

    # tant qu'on n'a pas de solution
    while True:
        Nbr_gene +=1

        # on selectionne les chemins (on modifie le labyrinthe aussi en posant des phéromones)
        tableau_p, Ne, fitness_moy, maze = selection(tableau_p.copy(), ts, maze.copy())

        # on regarde si la end cell du premier chemin du tableau trié pointe sur la case d'arrivée
        i, j, foo = end_cell(tableau_p[0], maze, coord_depart)
        if maze[i][j] == 2:
            break

        # on reproduit notre population et on la fait muter
        tableau_p = reproduction(tableau_p.copy(), Ne)
        tableau_p = mutation(tableau_p.copy(), tm)

        gen.append(Nbr_gene)
        value_moy.append(fitness_moy)
        # print(fitness_moy)
        # print(tableau_p[0])

    # on a trouvé un chemin
    index = 0
    pos_i, pos_j = coord_depart
    pos_i, pos_j = suivant(tableau_p[0][index], pos_i, pos_j)

    # affiche le graphique de fitness / gene
    plt.plot(gen, value_moy, 'r')
    plt.xlabel("generation")
    plt.ylabel("fitness moy")
    plt.title("fonction de fitnesse en fonction de la generation")
    plt.show()


    # modifie le labyrinthe (0 -> 4) avec le chemin qui  la solution
    while maze[pos_i][pos_j] != 2:
        if maze[pos_i][pos_j] != 3:     # pour ne pas écrire par dessus la case de départ si il y a des allers-retours
            maze[pos_i][pos_j] = 4
        index += 1
        direction = tableau_p[0][index]
        pos_i, pos_j = suivant(direction, pos_i, pos_j)

    print("nbr generations:",Nbr_gene, "\nchemin:",tableau_p[0],"\n\n\n")
    return maze, Nbr_gene


def ouvrir_labyrinthe(string):
    """" ouvre un labyrinthe à partir d'un fichier texte"""

    tableau = []
    labyrinthe = []
    with open(string) as input_file:
        for line in input_file:
            tableau.append(line[:-1])

    for i in range(len(tableau)):
        labyrinthe.append([])
        for elem in tableau[i]:
            labyrinthe[i].append(int(elem))

    return labyrinthe


################################################ autre #################################################################


def next_case(maze, pos_i, pos_j, pos_i_pre, pos_j_pre):
    """ sous fonction de resol_other_way, prend en entré un labyrinthe, les coord de la case sur laquelle on est et
        celle de la case précédente.
        ressort une liste avec les cases adjacentes sur lesquelles onn peut se déplacer, sans la case d'où on vient"""

    cod_i = pos_i
    cod_j = pos_j

    coord = []
    for ligne in range(3):  # parcourt i,j et le carré de 3 * 3 autour de i,j
        for colonne in range(3):
            if not (pos_i + 1 - ligne == cod_i and pos_j + 1 - colonne == cod_j):
                # si les coordonnées sont dans le tableau
                if 0 <= (pos_i + 1 - ligne) < len(maze) and 0 <= (pos_j + 1 - colonne) < len(maze):
                    # si la case correpondante à ces coordonnées vaut 1,2 ou 3 (chemin libre, case arrivée, case depart)
                    if maze[pos_i + 1 - ligne][pos_j + 1 - colonne] in (1, 2, 3) and (
                    pos_i + 1 - ligne, pos_j + 1 - colonne) != (pos_i_pre, pos_j_pre):
                        # on incrémente le compteur
                        coord.append((pos_i + 1 - ligne, pos_j + 1 - colonne))

    return coord


def resol_other_way(maze):
    """ resoud un labyrinthe pas de façon génétique, prend un labyrinthe sans but et sans goal et
        ressort le labyrinthe avec le chemin . S'appuie sur différentes sous fonctions"""

    # trouve les coord de depart et du goal
    for i in range(len(maze)):
        for j in range(len(maze)):
            if maze[i][j] == 3:
                coord_depart = (i, j)
            elif maze[i][j] == 2:
                coord_goal = (i, j)


    #print(maze.copy()) # affiche le labyrinthe dans le terminal

    Nbr_gene = 0            # calcule le nombre de generations

    chemin = [coord_depart,coord_depart]

    value_moy = []  # tableau qui prend les valeurs de fitness moyennes (pour tracer une courbe)
    gen = []        # tableau qui prend le nombre de generations (pour tracer une courbe)

    # initialisation, on part de la case de départ
    pos_i_pre = coord_depart[0]
    pos_i = coord_depart[0]
    pos_j_pre = coord_depart[1]
    pos_j = coord_depart[1]

    # tant qu'on n'est pas arrivé sur la case goal
    while True:
        Nbr_gene +=1

        # on met dans un tableau les cases sur lesquelles on peut se déplacer (sans mettre la case d'où on vient)
        next_cases = next_case(maze,pos_i,pos_j,pos_i_pre,pos_j_pre)

        # si il y a une ou des cases dans le tableau, on choisi celle qui nous rapproche le plus de la case goal
        if next_cases:
            value_min = len(maze) ** 2
            next_coord = next_cases[0]
            for coord in next_cases:
                # if maze[coord[0]][coord[1]]==2:
                #     break
                value = dist(coord, coord_goal)
                if value < value_min:
                    value_min = value
                    next_coord = coord

            chemin.append(next_coord)
            pos_i_pre = pos_i
            pos_j_pre = pos_j
            pos_i, pos_j = next_coord

        # s'il n'y a pas de case dans le tableau cela veut dire que l'on est dans une impasse
        # on rebrousse chemin jusqu'au prochain embranchement en coloriant les cases pour ne pas y retourner
        else:
            while not next_cases and maze[pos_i][pos_j] not in (2, 3):
                # alors on continue à poser une phéromone et on passe à la case d'après
                maze[pos_i][pos_j] = 5
                chemin.pop()
                pos_i_pre,pos_j_pre = chemin[-2]
                pos_i, pos_j = chemin[-1]
                next_cases = next_case(maze,pos_i,pos_j,pos_i_pre,pos_j_pre)
                # tant que la nouvelle case est déja colorée (si retour en arrière) on prend la suivante

        # si on est arrivé sur la case goal, on sort du while
        if maze[pos_i][pos_j] == 2:
            break

        #print(chemin)
        gen.append(Nbr_gene)
        value_moy.append(value_min)
        #print(value_min)

    # on a trouvé un chemin

    # affiche le graphique de fitness / gene
    plt.plot(gen, value_moy, 'r')
    plt.xlabel("generation")
    plt.ylabel("distance au goal")
    plt.title("distance au goal en fonction de la generation (autre algo)")
    plt.show()

    # modifie le labyrinthe (0 -> 4) avec le chemin qui  la solution
    for coord in chemin:
        pos_i, pos_j = coord
        # pour ne pas écrire par dessus la case de départ/goal
        if maze[pos_i][pos_j] != 3 and maze[pos_i][pos_j] != 2:
            maze[pos_i][pos_j] = 4


    print("nbr generations:",Nbr_gene, "\nchemin:",chemin,"\n\n\n")
    return maze, Nbr_gene


#################################################   main   #############################################################
# labyrinthe = generate_maze(10)                               # génération du labyrinthe
# labyrinthe = add_goal_and_start_points_for_algo_gene(labyrinthe.copy()) # ajout d'un but

# ouvre et affiche 2 labyrintes spécifiques à partir de fichiers textes
labyrinthe = ouvrir_labyrinthe('test4.txt')
display_maze(labyrinthe.copy(), "labyrinthe")  # affichage du labyrinthe
debut1 = time()
labyrinthe_res, b= resol_maze_genetic(labyrinthe.copy(), 0.5, 0.5)  #(maze, ts, tm)
temps1 = time() - debut1
display_maze(labyrinthe_res,"resolution algo génétique en {} s".format(round(temps1,3))) # affichage du labyrinthe

labyrinthe = ouvrir_labyrinthe('test2.txt')
display_maze(labyrinthe.copy(), "labyrinthe")  # affichage du labyrinthe
debut1 = time()
labyrinthe_res, b = resol_maze_genetic(labyrinthe.copy(), 0.5, 0.5)  #(maze, ts, tm)
temps1 = time() - debut1
display_maze(labyrinthe_res,"resolution algo génétique en {} s".format(round(temps1,3))) # affichage du labyrinthe

# genere 10 labyrinthes de taille 10, affiche les labyrinthes et courbes
# ne pas oublier de fermer les onglets sinon ça ne continue pas
gene = 0
for i in range(10):

    labyrinthe = generate_maze(10)
    labyrinthe = add_goal_and_start_points_for_algo_gene(labyrinthe.copy())
    display_maze(labyrinthe.copy(), "labyrinthe")
    labyrinthe_res, gen = resol_maze_genetic(labyrinthe.copy(), 0.5, 0.5)  # (maze, ts, tm)
    display_maze(labyrinthe_res, "resultat")
    gene+=gen

print("fin, generation moyenne pour 10 labyrinthes",gene/10)


gene = 0
for i in range(10):

    labyrinthe = generate_maze(100)
    labyrinthe = add_goal_and_start_points_for_algo_gene(labyrinthe.copy())
    display_maze(labyrinthe.copy(), "labyrinthe")
    labyrinthe_res, gen = resol_other_way(labyrinthe.copy())
    display_maze(labyrinthe_res, "resultat pas algo genetique")
    gene+=gen

print("fin, generation moyenne pour 10 labyrinthes",gene/10)