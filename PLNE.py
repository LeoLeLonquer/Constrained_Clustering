#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import numpy as np
from  docplex.mp import model
from  sklearn.datasets import make_moons, make_blobs
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import collections
from itertools import combinations, product
import propagation 

# HERE WE ARE GONNA TRY TO MINIMIZE THE MAXIMUM DIAMETER OF  CLUSTERS

# Two differents ways to implement the pb
# 1 : using y(i,j) where y(i,j)==1 iff i and j are in the same cluster
# 2 : using x(i,k) where x(i,k)==1 iff i is in cluster k



# if x1==x2 and x2==x3 then x1==x3
# Is it better to preprocess the constraint
# Or to add constraints
# Or to do nothing ?
                              # still work when [1,-1] is in eq_cons
#================================================================================
# APPROACH 1

# Start to modelize
def model1():
    with model.Model('Minimizing max diameter 1') as pb :

        y = pb.integer_var_list(n_samples**2, 0, 1, 'y')
        maxD = pb.integer_var()

        #TODO
        # add constraint between the y[i]
        for i,dist in enumerate(flat_dist):
            pb.add_constraint(maxD >= dist*y[i])

        pb.set_objective('min', maxD)

        sol = pb.solve()
        # print(sol)

        ysol = np.array([sol.get_value(yvar) for yvar in y]).reshape(n_samples, n_samples)
        # print(ysol)
#================================================================================
# APPROACH 2

def cons_linking(cons):
    dico = collections.defaultdict(set) # pt => list of pts
    # linked = collections.defaultdict(set) # nb => list of pt
    asso = dict() # pt => nb
    for pt1, pt2 in cons:
        dico[pt1].add(pt2)
        dico[pt1].add(pt1)
        dico[pt2].add(pt1)
        dico[pt2].add(pt2)
        asso[pt1] = pt1
        asso[pt2] = pt2

    pts = set(dico.keys())
    for pt in pts:
        for pt2 in pts :
            # if the intersection is not empty
            if asso[pt] != asso[pt2] \
            and not not dico[asso[pt]].intersection(dico[asso[pt2]]):
                dico[asso[pt]].update(dico[asso[pt2]])
                asso[pt2] = pt
                del dico[pt2]

    return list(dico.values())

# Start to modelize
def PLNE(data,  ml_cons, cl_cons, nb_cluster):
    distance_matrix = pairwise_distances(data) # D[i,j] is the distance between i and j
    flat_dist = distance_matrix.flatten()
    pb = model.Model('Minimizing max diameter 2')
    tstart = time.time()
    # Adapting the variables to the current one
    n_samples = len(data)
    Klusters = nb_cluster

    # Defining the global problem
    pb = model.Model('Minimizing max diameter')
    x = np.array(pb.integer_var_list(n_samples*Klusters, 0, 1, 'x'))
    x = x.reshape(n_samples, Klusters)

    maxD = pb.integer_var()
    pb.add_constraints([sum(xl) == 1 for xl in x])

    # add the sum of column must be sup or equal to 1
    for i in range(n_samples):
        for j in range(n_samples):
            for k in range(Klusters):
                pb.add_constraint(maxD >= distance_matrix[i,j]
                        *(x[i,k] + x[j,k] - 1))

    # Defining new constraints
    # new_cons = cons_linking(eq_constraints)
    ml_groups, cl_groups = propagation.propagate(n_samples,
                                                ml_cons,
                                                cl_cons)

    # MUST LINK CONSTRAINTS
    for cons in ml_groups :
        for pt1, pt2 in combinations(cons, 2):
            for k in range(Klusters):
                pb.add_constraint(x[pt1,k] == x[pt2,k])

    # CANNOT LINK CONSTRAINTS
    for label, not_linking_labels in cl_groups.items():
        for other_label in not_linking_labels :
            for pt1, pt2 in product(list(ml_groups[label]), list(ml_groups[other_label])):
                for k in range(Klusters):
                    pb.add_constraint((x[pt1,k] + x[pt2,k]) <= 1)

    pb.set_objective('min', maxD)
    sol = pb.solve()
    tend = time.time()
    duration = tend - tstart

    if sol == None :
        out = (np.full(n_samples, np.nan), np.nan, duration)
    else :
        xsol = np.array([sol.get_value(xvar) for xvar in x.flatten()]).reshape(n_samples, Klusters)
        colors = np.full(n_samples, np.nan)
        for l, pt_l in enumerate(xsol) :
            for k, pt_c in enumerate(pt_l):
                if int(pt_c) == 1:
                    colors[l] = k
        score = sol.get_value(maxD)
        out = (colors, score, duration)
    return duration,colors
    # if sol == None:
        # print("No solution found")
    # else:
        # xsol = np.array([sol.get_value(xvar) for xvar in x.flatten()]).reshape(n_samples, Klusters)

        # for k in range(Klusters):
            # xmask = xsol[:,k]==1
            # Xprime = X[xmask]
            # plt.plot(Xprime[:,0], Xprime[:,1], 'x')

        # # Plotting the constraints
        # # Plotting the equality constraints
        # for cons in new_cons:
            # new_X_x = []
            # new_X_y = []
            # for pt in cons :
                # new_X_x.append(X[pt, 0])
                # new_X_y.append(X[pt, 1])
            # plt.plot(new_X_x,new_X_y,
                     # '^', linewidth=50)

        # for pt1, pt2 in cl_cons:
            # plt.plot([X[pt1,0],X[pt2,0]],
                     # [X[pt1,1],X[pt2,1]],
                     # 'v', linewidth=50)
        # plt.show()

