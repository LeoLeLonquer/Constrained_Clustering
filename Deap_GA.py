from deap import base, creator
import random
from deap import tools
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.metrics.pairwise import pairwise_distances
from sklearn import datasets

import propagation

def call_GA(data,ml_cons=[],cl_cons=[],nb_cluster=3):
    Dico_fortement_connexe,cl_rules=propagation.propagate(len(data), ml_cons, cl_cons)
    IND_SIZE = len(Dico_fortement_connexe) # Number of connex sets
    POP_SIZE = 100

    Dists = pairwise_distances(data)

    #Create score
    creator.create("FitnessDS", base.Fitness, weights=(-1.0,1.0)) #minimize D, maximize S
    creator.create("Individual", list, fitness=creator.FitnessDS)

    #Defining population
    toolbox = base.Toolbox()
    toolbox.register("affectation", random.randint,1,nb_cluster)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.affectation, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def nb_broken_rules(individual):
        broken_rules=0
        for i in range(len(individual)):
            for j in range(i,len(individual)):
                if i in cl_rules and j in cl_rules[i] and individual[i]==individual[j]:
                    broken_rules += 1
        return broken_rules # 1 wrong association can cause multible broken rules

    def evaluate(individual):
        #Diameter score: greatest distance beetween two items of the same cluster(to mnimize)
        D = 0
        S = None
        Total=0

        Attribs={}
        for k in range(1,nb_cluster+1):
            #Find all elements of the cluster k
            Global_set= [list(Dico_fortement_connexe[i]) for i in range(len(individual)) if individual[i] == k]
            Attribs[k] = [item for sublist in Global_set for item in sublist]

        for k in range(1,nb_cluster+1):
            for i in Attribs[k]:
                #check in same set for diameter
                for j in Attribs[k]: #redoing many symetric distances
                    #Find greatest dist
                    dist = Dists[i][j]
                    if dist > D:
                        D = dist
                """
                #Calculates S score
                #check in other set for Inter cluster dist
                for k2 in range(1,nb_cluster+1):
                    if k2 != k:
                        for o in Attribs[k2]:
                            #Find smallest dist
                            dist = Dists[i][o]
                            if S == None or dist < S:
                                S = dist
                            Total += dist
                """

        #Penality factor ( for minimizing).
        malus = (1+nb_broken_rules(individual))
        #return D * malus,Total / malus,
        return D * malus,0,


    toolbox.register("mate", tools.cxUniformPartialyMatched,indpb=0.5)
    #toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt,low=1,up=nb_cluster)
    #toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("select", tools.selRoulette)

    toolbox.register("evaluate", evaluate)



    def main():
        pop = toolbox.population(n=POP_SIZE) #set population size here
        CXPB, MUTPB, NGEN = 0.8, 0.8, 100
        #even number that is about half the population:
        HALF_SIZE=POP_SIZE//2 if POP_SIZE//2 % 2 == 0  else (POP_SIZE//2) + 1
        #HALF_SIZE=POP_SIZE-(POP_SIZE//4 if POP_SIZE//4 % 2 == 0  else (POP_SIZE//4) + (4-POP_SIZE%4))

        # Evaluate the entire population
        fitnesses = map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit


        maxfitness = tools.selBest(pop, k=1)[0].fitness.values[0]

        score_evo=[]
        tstart = time.time()
        tfin=tstart
        for g in range(NGEN):
            # Select the next generation individuals
            offspring = toolbox.select(pop,HALF_SIZE)
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring)) #added list

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant,indpb = max(0.05,1/(g+1)))
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is partialy replaced by the offspring
            pop=tools.selBest(pop, k=POP_SIZE-HALF_SIZE) + offspring


            if g % 500 == 0: #wipe out invalid pop
                pop = [ind for ind in pop if nb_broken_rules(ind) == 0]
                #print("OK INDIVIDUALS : ", len(pop))
                pop = pop + toolbox.population(n=POP_SIZE-len(pop))
                #print ("NEW_POP",len(pop))

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in pop if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit


            if g % 20 == 0:
                chicken_dinner=tools.selBest(pop, k=1)
                #print("Gen {} | rules broken : {}  | score {} ".format(g,nb_broken_rules(chicken_dinner[0]),chicken_dinner[0].fitness))
            s = tools.selBest(pop, k=1)[0].fitness.values[0] #doesn't check if individual is valid
            score_evo+=[s]
            if s < maxfitness:
                maxfitness=s
                print(maxfitness)
                tfin=time.time()-tstart

        return tfin,pop,score_evo

    tfin,endpop,se = main()

    plt.figure()
    plt.title("Score evolution")
    plt.plot(se)
    #blocks process
    plt.draw()

    chicken_dinner=tools.selBest(endpop, k=1)[0]
    print("==============================", chicken_dinner.fitness.values[0],
            nb_broken_rules(chicken_dinner))
    #dégroupage des points
    colors=np.zeros(len(data))
    for i in range(IND_SIZE):
        for j in Dico_fortement_connexe[i]:
            colors[j]=chicken_dinner[i]
    print(colors)
    return colors,chicken_dinner.fitness,tfin


datasize = 100
data = datasets.make_blobs(n_samples=datasize, centers= 3 ,random_state=8)[0]

Dists = pairwise_distances(data)
ml_cons=[(1,2),(13,11),(11,22)]
cl_cons=[(1,7),(12,29),(17,20)]
#ml_cons=[]
#cl_cons=[]

colors,best_score,tt = call_GA(data,ml_cons,cl_cons,3)
