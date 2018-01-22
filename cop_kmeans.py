#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 19:44:57 2018

@author: dwu
"""

# -*- coding: utf-8 -*-
import random

import matplotlib.pyplot as plt

import sys
import argparse

def cop_kmeans(dataset, k, ml=[], cl=[], 
               initialization='kmpp', 
               max_iter=300, tol=1e-4):
    ml, cl = transitive_closure(ml, cl, len(dataset))
    ml_info = get_ml_info(ml, dataset)
    tol = tolerance(tol, dataset)
    centers = initialize_centers(dataset, k, initialization)
    clusters = [-1] * len(dataset)
    for i in range(max_iter):
        clusters_ = [-1] * len(dataset)
        for i, d in enumerate(dataset):
            indices, _ = closest_clusters(centers, d)
            counter = 0
            if clusters_[i] == -1:
                found_cluster = False
                while (not found_cluster) and counter < len(indices):
                    index = indices[counter]
                    violate_contraint = []
                    if not violate_constraints(i, index, clusters_, ml, cl):
                        found_cluster = True
                        clusters_[i] = index
                        for j in ml[i]:
                            clusters_[j] = index
                    counter += 1
                if not found_cluster:
                    print ("problem id {}".format(i))
                    print ("- - - - - -  no found cluster - - - - - - ")
                    return [],[]
        
        clusters_, centers_ = compute_centers(clusters_, dataset, k, ml_info)
        shift = sum(l2_distance(centers[i], centers_[i]) for i in range(k))
        if shift <= tol:
            break
        clusters = clusters_
        centers = centers_
    
    return clusters_, centers_




def l2_distance(point1, point2):
    return sum([(float(i)-float(j))**2 for (i, j) in zip(point1, point2)])

# taken from scikit-learn (https://goo.gl/1RYPP5)
def tolerance(tol, dataset):
    n = len(dataset)
    dim = len(dataset[0])
    averages = [sum(dataset[i][d] for i in range(n))/float(n) for d in range(dim)]
    variances = [sum((dataset[i][d]-averages[d])**2 for i in range(n))/float(n) for d in range(dim)]
    return tol * sum(variances) / dim

def closest_clusters(centers, datapoint):
    distances = [l2_distance(center, datapoint) for
                 center in centers]
    return sorted(range(len(distances)), key=lambda x: distances[x]), distances

def initialize_centers(dataset, k, method):
    if method == 'random':
        ids = list(range(len(dataset)))
        random.shuffle(ids)
        return [dataset[i] for i in ids[:k]]        
    
    elif method == 'kmpp':
        chances = [1] * len(dataset)
        centers = []
        
        for _ in range(k):
            chances = [x/sum(chances) for x in chances]        
            r = random.random()
            acc = 0.0
            for index, chance in enumerate(chances):
                if acc + chance >= r:
                    break
                acc += chance
            centers.append(dataset[index])
            
            for index, point in enumerate(dataset):
                cids, distances = closest_clusters(centers, point)
                chances[index] = distances[cids[0]]
                
        return centers

def violate_constraints(data_index, cluster_index, clusters, ml, cl):
    for i in ml[data_index]:
        if clusters[i] != -1 and clusters[i] != cluster_index:
            return True


    for i in cl[data_index]:
        if clusters[i] == cluster_index:
            return True
    return False

def compute_centers(clusters, dataset, k, ml_info):
    cluster_ids = set(clusters)
    k_new = len(cluster_ids)
    id_map = dict(zip(cluster_ids, range(k_new)))
    clusters = [id_map[x] for x in clusters]    
    
    dim = len(dataset[0])
    centers = [[0.0] * dim for i in range(k)]

    counts = [0] * k_new
    for j, c in enumerate(clusters):
        for i in range(dim):
            centers[c][i] += dataset[j][i]
        counts[c] += 1
        
    for j in range(k_new):
        for i in range(dim):
            centers[j][i] = centers[j][i]/float(counts[j])

    if k_new < k:
        ml_groups, ml_scores, ml_centroids = ml_info
        current_scores = [sum(l2_distance(centers[clusters[i]], dataset[i]) 
                              for i in group) 
                          for group in ml_groups]
        group_ids = sorted(range(len(ml_groups)), 
                           key=lambda x: current_scores[x] - ml_scores[x],
                           reverse=True)
        
        for j in range(k-k_new):
            gid = group_ids[j]
            cid = k_new + j
            centers[cid] = ml_centroids[gid]
            for i in ml_groups[gid]:
                clusters[i] = cid
                
    return clusters, centers
    
def get_ml_info(ml, dataset):
    flags = [True] * len(dataset)
    groups = []
    for i in range(len(dataset)):
        if not flags[i]: continue
        group = list(ml[i] | {i})
        groups.append(group)
        for j in group:
            flags[j] = False
    
    dim = len(dataset[0])
    scores = [0.0] * len(groups)
    centroids = [[0.0] * dim for i in range(len(groups))]
    
    for j, group in enumerate(groups):
        for d in range(dim):
            for i in group:
                centroids[j][d] += dataset[i][d]
            centroids[j][d] /= float(len(group))

    scores = [sum(l2_distance(centroids[j], dataset[i])
                  for i in groups[j]) 
              for j in range(len(groups))]
    
    return groups, scores, centroids

def transitive_closure(ml, cl, n):
    ml_graph = dict()
    cl_graph = dict()
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        add_both(ml_graph, i, j)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
    for (i, j) in cl:
        add_both(cl_graph, i, j)
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)
    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise Exception('inconsistent constraints between %d and %d' %(i, j))
    return ml_graph, cl_graph
    
def max_D_one_cluster(data,label) :
    dis_max=0
    point = []
    for i in range(len(label)):
        for j in range(i,len(label)) : 
            if l2_distance(data[i],data[j])> dis_max :
                dis_max=l2_distance(data[i],data[j])
                point = [i,j]
    return dis_max,point

def max_D_all_cluster(data,label,k):
    distance_max =-1
    cluster_dis_max = -1
    point = [-1,-1]
    for i in range(k):
        index = [x for x,y in enumerate(label) if y==i]
        distance,points = max_D_one_cluster(data,index)
        if (distance_max<distance) : 
            distance_max = distance
            cluster_dis_max = i
            point = points
    return distance_max,point,[cluster_dis_max,cluster_dis_max]

def get_score(data,label,k):
    distance,_,_=max_D_all_cluster(data,label,k)
    return distance

def min_D_between_cluster(data,label):
    dis_min=10000000000
    point = []
    cluster = []
    for i in range(len(data)):
        for j in range(i,len(data)):
            if label[i]!=label[j] and l2_distance(data[i],data[j])<dis_min :
                point=[i,j]
                cluster = [label[i],label[j]]
                dis_min = l2_distance(data[i],data[j])
    return dis_min,point,cluster
#!/usr/bin/env python
# -*- coding: utf-8 -*-


def calcule_score(type_score,data,label,k):
    if type_score == 'D' :
        return max_D_all_cluster(data,label,k)
    elif type_score =='S':
        return min_D_between_cluster(data,label)

def rand_score(res , label):
    n = len(label)
    a = 0
    b = 0
    if len(res) != len(label) :
        print ("problem nb donnee")
        return None
    for i in range(n):
        for j in range(i,n):
            if res[i]==res[j] and label[i]==label[j]:
                a+=1
            if res[i]!=res[j] and label[i]!=label[j]:
                b+=1
    
    return 2*(a+b)/(n*(n-1))
### import matplotlib.pyplot as plt
import time
# plot_data=[[x,y] for x,y in zip(a,b)]
def run_cop_kmeans (data,ml,nl,k):
    start = time.time()
    best_clusters = None
    best_score = None    
    for i in range(10):
        clusters, centers = cop_kmeans(data, k,ml,nl, max_iter=10,tol=0.1)
    #         score = sum(l2_distance(data[j], centers[clusters[j]]) 
    #                     for j in range(len(data)))
        score,_,_ =calcule_score('D',data,clusters,k) 
        print("score = {}".format(score))
        if best_score is None or score > best_score:
            best_score = score
            best_clusters = clusters
            best_center=centers
    
    t = time.time()-start
    color = ['b', 'r', 'r', 'c', 'm', 'y', 'k']
    
    print (best_center)
    for  i in range(k):
        index =  [x for x,y in enumerate(best_clusters) if y==i]
        plt.scatter([data[x][0] for x in index],[data[y][1] for y in index],color=color[i],marker = '.')
    # plt.scatter([data[x][0] for x in index],[data[y][1] for y in index],color=color[i],marker = '.')
    plt.show()
    return t,best_clusters
    