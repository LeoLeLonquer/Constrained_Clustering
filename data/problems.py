import numpy as np

# pour sauvegarde et lecture dans les fichiers
import pickle
from inspect import currentframe, getframeinfo
from pathlib import Path
from sklearn import datasets
import math
import random


datasize = 100

#####  LECTURE/ECRITURE DANS LES FICHIERS/ GENERATION DE DONNEES #####

# on obtient le nom du répertoire courant
fich_courant = getframeinfo(currentframe()).filename
rep_courant = Path(fich_courant).resolve().parent

def creer_donnees() :

    dat = {}
    dat["2blobs"] = {}
    dat["2blobs"]["data"], dat["2blobs"]["labels"] = datasets.make_blobs(n_samples=datasize, centers=2 ,random_state=8)
    # dat["2blobs"]["mustlink"] =
    # dat["2blobs"]["cannotlink"] =

    dat["3blobs"] = {}
    dat["3blobs"]["data"], dat["3blobs"]["labels"] = datasets.make_blobs(n_samples=datasize, centers=3 ,random_state=8)
    # dat["3blobs"]["mustlink"] =
    # dat["3blobs"]["cannotlink"] =
    dat["2moons"] =  {}
    dat["2moons"]["data"], dat["2moons"]["labels"] = datasets.make_moons(n_samples=datasize, noise=.05)
    # dat["2moons"]["mustlink"] =
    # dat["2moons"]["cannotlink"] =

    return dat

# écriture de données dans un fichier
def ecriture(donnees, nom_fichier):
    with open(str(rep_courant) + "/" + str(nom_fichier),"wb") as fichier :
        pick = pickle.Pickler(fichier)
        pick.dump(donnees)

# lecture des donnees depuis le chemin indiqué
def lecture(nom_fichier):
    with open(str(rep_courant) + "/" + str(nom_fichier),"rb") as fichier :
        depick = pickle.Unpickler(fichier)
        dat = depick.load()
    return dat

def creer_contraintes (label, pourcent) :
    mustlink = []
    cannotlink = []
    nb_points = 0
    
    # calcul du nombre de points à contraindre
    if (0 < pourcent) and (pourcent <= 1) :    
        nb_points = math.floor(pourcent * len(label))
        # il faut un nombre pair de points, ils vont deux par deux
        if (nb_points % 2 == 1) :
            nb_points = nb_points - 1
    
    # les indices des points contraints
    echantillon = random.sample(range(1, len(label)), nb_points)

    # division en deux groupes d'echantillon
    milieu = int(len(echantillon)/2)
    half_1 = echantillon[:milieu]
    half_2 = echantillon[milieu:]
    # creation contraintes
    for i in range(milieu) :
        if (label[half_1[i]] == label[half_2[i]]) :
            mustlink.append((half_1[i],half_2[i]))
        else :
            cannotlink.append((half_1[i],half_2[i]))

    return mustlink, cannotlink

dat = creer_donnees()
ecriture(dat,"data.dat")
data_lu = lecture("data.dat")
data = data_lu["2moons"]["data"]
ml,cl = creer_contraintes(data_lu["2moons"]["labels"], 0.20)
