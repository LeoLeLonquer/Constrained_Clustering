#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from  docplex.mp import model
from  sklearn.datasets import make_moons, make_blobs
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import collections
from itertools import combinations
from propagation import propagate

# HERE WE ARE GONNA TRY TO MINIMIZE THE MAXIMUM DIAMETER BETWEEN CLUSTERS

# Two differents ways to implement the pb
# 1 : using y(i,j) where y(i,j)==1 iff i and j are in the same cluster
# 2 : using x(i,k) where x(i,k)==1 iff i is in cluster k

function = make_blobs
# Get the data, compute the distance matrix
n_samples = 100
seed = 10
X, _Y = function(n_samples=n_samples,
                 random_state=seed) # X array of [x1,x2] and Y is a label

distance_matrix = pairwise_distances(X) # D[i,j] is the distance between i and j
flat_dist = distance_matrix.flatten()

# Set the number of clusters
Klusters = 3
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
def model2():
    with model.Model('Minimizing max diameter 2') as pb :
        # TODO
        # might seem to have some problem with the constraints
        # that are not taken into account sometimes
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
        #eq_constraints = []

        # if x1==x2 and x2==x3 then x1==x3
        # Is it better to preprocess the constraint
        # Or to add constraints
        # Or to do nothing ?
        eq_constraints = []
        eq_constraints = [(1,-1),
                          (1,3),
                          (3,2),
                          (4,-3),
                          (4,6),
                          (7,8),
                          ]

        new_cons = cons_linking(eq_constraints)
        for cons in new_cons :
            for pt1, pt2 in combinations(cons, 2):
                for k in range(Klusters):
                    pb.add_constraint(x[pt1,k] == x[pt2,k])

        # for pt1, pt2 in eq_constraints:
            # for k in range(Klusters):
                # pb.add_constraint(x[pt1,k]==x[pt2,k])

        noneq_constraints = []
        noneq_constraints = [(51, 50),
                             (51, 30),
                             (3, 11)
                            ]
                                      # TODO to test that adding [1,-1]
                                      # still work when [1,-1] is in eq_cons

        ml_groups, cl_cons = propagate(n_samples, eq_constraints, noneq_constraints)
        print(ml_groups, dict(cl_cons))

        for pt1, pt2 in noneq_constraints:
            for k in range(Klusters):
                pb.add_constraint((x[pt1,k] and x[pt2,k]) == 0)

        pb.set_objective('min', maxD)

        sol = pb.solve()

        xsol = np.array([sol.get_value(xvar) for xvar in x.flatten()]).reshape(n_samples, Klusters)

        # for cons in new_cons:
            # print("=========")
            # for pt in cons:
                # print(xsol[pt,:])

        for k in range(Klusters):
            xmask = xsol[:,k]==1
            Xprime = X[xmask]
            plt.plot(Xprime[:,0], Xprime[:,1], 'x')

        # Plotting the constraints
        # Plotting the equality constraints
        for cons in new_cons:
            new_X_x = []
            new_X_y = []
            for pt in cons :
                new_X_x.append(X[pt, 0])
                new_X_y.append(X[pt, 1])

            plt.plot(new_X_x,new_X_y,
                     '^', linewidth=50)

        # for pt1, pt2 in eq_constraints:
            # plt.plot([X[pt1,0],X[pt2,0]],
                     # [X[pt1,1],X[pt2,1]],
                     # 'x', linewidth=50)

        for pt1, pt2 in noneq_constraints:
            plt.plot([X[pt1,0],X[pt2,0]],
                     [X[pt1,1],X[pt2,1]],
                     'v', linewidth=50)
        plt.show()

model2()
