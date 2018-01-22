import collections
import numpy as np
from  sklearn.datasets import make_moons, make_blobs
# AG
# PLNE
# CSP
# KMeans


PB_SIZES = [10, 100, 10**3, 10**4, 10**5, 10**6]
# MUST_LINK AND CANNOT_LINK CONSTRAINT DESCR :
# element : (pourcentage of pts to connect,
#           range of connection per point)
MUST_LINK = [(0, [0,0]), (1,[1,2]), (10, [1,5]),(50, [1,10]),(100,[1,20])]
CANNOT_LINK = [(0, [0,0]), (1,[1,2]), (10, [1,5]),(50, [1,10]),(100,[1,20])]
PB_FUNCTION = [make_moons, make_blobs]
SEED = [1,2,3]

def create_pb(pb_size, ml_cond, cl_cond, pb_fun, seed=1):
    data, labels = pb_fun(n_samples=pb_size, random_state=seed)
    labels_groups = create_label_groups(labels)
    must_link = create_ml_cons(labels, labels_groups, ml_cond)
    cannot_link = create_cl_cons(labels, labels_groups, cl_cond)

    return data, must_link, cannot_link


def create_label_groups(labels):
    labels_groups = collections.defaultdict(set)
    for index, y in enumerate(labels):
        labels_groups[y].add(index)
    return(labels_groups)

def create_ml_cons(labels, labels_groups, cons):
    percentage, ranged = cons
    nb_co = round((percentage/100) * len(labels))
    min_co, max_co = ranged[0], ranged[1]
    nb_points = len(labels)
    new_cons = []

    while nb_co > 0 : # add a cond if no possibility to link
        cos = np.random.randint(min_co, max_co)
        pt = np.random.randint(0, nb_points)
        linked_pt = set({})
        cos_bis = 0

        for co in range(1, cos + 1):
            if len(labels_groups[labels[pt]]) > 1:
                pt_to_co = labels_groups[labels[pt]].pop()
                while pt_to_co == pt :
                    pt_to_co = labels_groups[labels[pt]].pop()
                linked_pt.add(pt_to_co)
                new_cons.append((pt, pt_to_co))
                cos_bis = cos_bis + 1
        nb_co = nb_co - cos
        while not not linked_pt:
            labels_groups[labels[pt]].add(linked_pt.pop())
    return new_cons

def create_cl_cons(labels, labels_groups, cons):
    percentage, ranged = cons
    nb_co = round((percentage/100) * len(labels))
    min_co, max_co = ranged[0], ranged[1]
    nb_points = len(labels)
    new_cons = []
    min_kluster, max_kluster = min(labels), max(labels)

    while nb_co > 0 : # add a cond if no possibility to link
        cos = np.random.randint(min_co, max_co)
        pt = np.random.randint(0, nb_points)
        linked_pt = collections.defaultdict(set)
        cos_bis = 0

        for co in range(1, cos + 1):
            kluster_to_co = np.random.randint(min_kluster, max_kluster)
            while labels[pt] == kluster_to_co:
                kluster_to_co = np.random.randint(min_kluster, max_kluster)
            if len(labels_groups[kluster_to_co]) > 1:
                pt_to_co = labels_groups[kluster_to_co].pop()
                while pt_to_co == co :
                    pt_to_co = labels_groups[kluster_to_co].pop()
                linked_pt[kluster_to_co].add(pt_to_co)
                new_cons.append((pt, pt_to_co))
                cos_bis = cos_bis + 1
        nb_co = nb_co - cos
        for key in linked_pt.keys():
            while not not linked_pt[key]:
                labels_groups[labels[key]].add(linked_pt[key].pop())
    return new_cons
