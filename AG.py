print("AG version 1.0")

import random as r
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import silhouette_score as sc
from sklearn.metrics import accuracy_score as accscore
from sklearn.metrics import v_measure_score as v_measure_score
from sklearn.metrics import calinski_harabaz_score as chscore
import math as m
from time import time

from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

# pour sauvegarde et lecture dans les fichiers
import pickle
from inspect import currentframe, getframeinfo
from pathlib import Path


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

dat = creer_donnees()
ecriture(dat,"data.dat")
data_lu = lecture("data.dat")
data = data_lu["2moons"]["data"]

##### FIN FONCTIONS LECTURE/ ECRITURE #####

# génération de points aléatoires
def gen_n_points(n,accu=[]):
    for i in range(n):
        accu +=[[r.randint(0,100),r.randint(0,100)]]
    Dists=[[ distance.euclidean(accu[i],accu[j]) for i in range(n)]for j in range(n)]
    return np.asarray(accu),np.asarray(Dists)

# trouve le meilleur point d'une population
def best(pop):
    return max(pop, key=lambda x:x['s'])

# dessine la solution
def plot_solution(data, labels,title):
    # colors = set(labels)
    plt.figure()
    plt.title(title)
    plt.scatter(data[:,0], data[:,1], c = labels)
    #blocks process
    plt.draw()

# dessine la meilleure solution
def plot_best_solution(data, pop, title = "Pensez à mettre un titre ;)"):
    best_ind = best(pop)
    plot_solution(data,best_ind["x"],title)

#Sandbox

# data,D = gen_n_points(datasize)
# data = datasets.make_blobs(n_samples=datasize, centers=3 ,random_state=8)[0]
# data = datasets.make_circles(n_samples=datasize, factor=.5,noise=.05)[0]


# Génération de datasets 
# data, targets = datasets.make_moons(n_samples=datasize, noise=.05)
# calcul des distances entre les points
dists = [[ distance.euclidean(data[i],data[j]) for i in range(len(data))]for j in range(len(data))]

nbvoisins = 10

# liste distance 5 voisins les plus proches
def make_dic(data,D):
    max_dist = np.amax(D)
    Dict={}
    for i in range(len(data)):
        ld= [(j,D[i][j]/max_dist) for j in range(len(D[i]))]
        del ld[i]

        Dict[i] = sorted(ld, key=lambda p: p[1])[:nbvoisins]
    return Dict

# précalcul des plus proches voisins de chaque point 
Dict = make_dic(data,dists)

# print(Dict[0])
# print(Dict[0][0])
# print(Dict[0][0][0])



#blobs = datasets.make_blobs(n_samples=datasize, random_state=8)
#no_structure = np.random.rand(datasize, 2), None

# génération aléatoire de centoïdes et d'affectations 
Centroids = [data[r.randint(0,datasize-1)] for i in range(3) ]
labels= [r.randint(0,3) for i in range(datasize)]

# number of clusters
k=2

# initialises a population without scores (aléatoirement)
def init_pop(n): 
    pop=[]
    for i in range(n):
        pop += [[r.randint(1,k) for i in range(datasize)]] # AN individual is a list of affectations to clusters
        #pop += [[1 for i in range(datasize)]] # zeros
    return pop


def moyenne_voisins(pop):
    memeclust = 0
    for i in range(len(pop)):
        for j in range(len(Dict[i])) :
            if pop[Dict[i][j][0]] == pop[i] :
                memeclust += 1 * 0.9**j
    return memeclust/len(pop)


# sets scores for individuals
def score(pop): 
    # définition de la fitness
    def fit(ind):
        # return sc(data, ind) if len(set(ind)) > 1 else -1
        # return accscore(targets, ind)
        # return v_measure_score(targets, ind)
        # print(chscore(data, ind))
        # return chscore(data, ind) if len(set(ind)) > 1 else -1
        # return distmin_inter_cluster(ind) if len(set(ind)) > 1 else -1
        return moyenne_voisins(ind) if len(set(ind)) > 1 else -1
    pop_score=[]
	# création de structure de données contenant le score
    for i in range(len(pop)):
        pop_score += [{"x":pop[i],"s":fit(pop[i])}]
    return pop_score

def distmin_inter_cluster(pop):

    distmin = -1
    
    # pour chaque affectation
    for i in range(len(pop)) :
        for j in range(len(pop)) :
            # si le point est d'un cluster différent du point actuel
            if pop[i] != pop[j] :
                # si la distance i-j est inférieure au min du cluster ou n'est pas initialisée
                if dists[i][j] < distmin or distmin < 0 :
                    distmin = dists[i][j]        
      
    return distmin


# Selects n best individuals for reproduction
def select(pop,n):#Selects individual for reproduction (roulette)
    accu=[]
    score_sum = sum([pop[i]["s"] for i in range(len(pop))])
    for i in range(n):#select an individual n times
        loto = r.uniform(0,score_sum)
        pin=0
        while loto > pop[pin]["s"]:
            loto -= pop[pin]["s"]
            pin +=1
        accu+=[pop[pin]]
    return accu

    pop_sorted = np.asarray(sorted(pop, key=lambda d: d['s'],reverse=True))
    return pop_sorted[:n]

# Select individuals to repopulate (la fonction KILL, on tue "child" fois)
def select2(pop, child): 
    pop_sorted = sorted(pop, key=lambda d: d['s'],reverse=True)
    return pop_sorted[:(len(pop)-len(child))] + child

# Performs crossover on a population (croisement des géniteurs)
def crossover(pop):
    def cross(i1,i2):
        n=r.randint(0,len(i1))
        return [i1["x"][:n] + i2["x"][n:], i2["x"][:n] + i1["x"][n:]]

    childs=[]
    for i in range(0,len(pop)-1,2):
        childs+= cross(pop[i],pop[i+1])
    return childs

# Mutates a population (apport de nouveaux allèles) 
def mutate(pop,mute_rate): #mutates a population (if an ind passes the mute rate test, we chose an arb number of affect to mutate)
    popm=pop
    ind_size=len(pop[0])
    for ind in range(len(popm)):#mutate every individual
            mutate_numbers=[r.randint(0,datasize-1) for i in range(int(datasize*mute_rate)+1) ] #choses labels to mutate
            for label in mutate_numbers:
                popm[ind][label]=r.randint(1,k)
    return popm

pop_size=100
# on génére une population et on lui attribue un score
pop = score(init_pop(pop_size)) #multiple of 2 pls
gen = 0
eps = 0.7
best_score=-1.0
nbneigh = 5

while gen < 1000:
# while best_score < eps and gen < 4000:

    print(gen, " ", best_score)

    if gen%100 == 0:
        print("Generation : ",gen)
        big_boss = best(pop)
        big_boss_score = big_boss["s"]
        if big_boss_score > best_score:
            best_score = big_boss_score
        #print("Our champion:\n",big_boss)
        #print("Our podium:",np.asarray(sorted(pop, key=lambda d: d['s'],reverse=True)))
    if gen%500 == 0:
        plot_best_solution(data,pop,"Gen : " + str(gen) + "  Score : " + str(big_boss_score) )#plots the best in the population

    # on séléctionne les géniteurs
    reprod = select(pop,pop_size//2) 

	# on génére une progéniture en croisant les géniteurs
    offspring = crossover(reprod)

	# on génére des mutants pour diversifier la pop avec nouveaux allèles
    offspring2 = mutate(offspring, (1/(gen + 1)))

	# scoring des mutants nouveaux nés (on nait sans score)
    offspring3 = score(offspring2)

	# on enlève les moins bons de la population 
    pop = select2(pop,offspring3)

    gen+=1

plot_best_solution(data,pop,"Gen : " + str(gen) + "  Score : " + str(big_boss_score) )#plots the best in the population
plt.show()
print("Ah vous êtes là vous!")
