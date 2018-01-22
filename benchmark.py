#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 19:47:33 2018

@author: dwu
"""

import random
import sys
import matplotlib.pyplot as plt
import time
from cop_kmeans import run_cop_kmeans,get_score
from pb_generation import create_pb
from  sklearn.datasets import make_moons, make_blobs
#from Deap_GA import call_GA

from maxD import model2

def k_means(data,ml,cl,k):
    return 1,[]
def PLNE(data,ml,cl,k) :
    return 1,[]
def CSP(data,ml,cl,k):
    return 1,[]
def ge(data,ml,cl,k):
    return 1,[]

pb = create_pb(200, (20,[0,2]), (20,[0,2]),make_blobs, seed=1)

#t,l = run_cop_kmeans(pb[0],pb[1],pb[2],5)

#print(t,l)

data = []
data.append(pb[0])
ml=[]
cl=[]
ml.append(pb[1])
cl.append(pb[2])


k=5
function = [run_cop_kmeans,model2,CSP,ge]
def run():
    pas_data=len(data)
    pas_contrainte=len(ml)
    res = [[{} for i in range(pas_data)] for j in range(pas_contrainte)]
    for t_data in range(pas_data):
        for t_con in range(pas_contrainte):
            test_data=data[t_data]
            test_ml =ml[t_con]
            test_cl =cl[t_con]
            for i in range(len(function)):
                t,label=function[i](test_data,test_ml,test_cl,k)
                score = get_score(test_data,label,k)
#                 print(t)
                res[t_data][t_con][i]=[t,score]
    return res
res = run()

plt.scatter([i for i in range(len(data))],[res[i][0][1][0] for i in range(len(data))],color='r',marker = '.')

plt.show()
