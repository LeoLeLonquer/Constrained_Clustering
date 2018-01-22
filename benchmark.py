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
from cop_kmeans import run_cop_kmeans
from pb_generation import create_pb
from Deap_GA import call_GA



def k_means(data,ml,cl,k):
    return 1
def PLNE(data,ml,cl,k) :
    return 1
def CSP(data,ml,cl,k):
    return 1
def ge(data,ml,cl,k):
    return 1

pb = create_pb(200, (20,[0,2]), (20,[0,2]),make_blobs, seed=1)

t,l = call_GA(pb[0],pb[1],pb[2],5)

print(t,l)

data = [1 for i in range(10)]
ml=[[] for i in range(10)]
cl=[[] for i in range(10)]

k=1
function = [run_cop_kmeans,PLNE,CSP,ge]
def run():
    pas_data=10
    pas_contrainte=10
    res = [[{} for i in range(pas_data)] for j in range(pas_contrainte)]
    for t_data in range(pas_data):
        for t_con in range(pas_contrainte):
            test_data=data[t_data]
            test_ml =ml[t_con]
            test_cl =cl[t_con]
            for i in range(len(function)):
                start = time.clock()
                score=function[i](test_data,test_ml,test_cl,k)
                time.sleep(0.001)
                t = time.clock()-start
#                 print(t)
                res[t_data][t_con][i]=[t,score]
    return res

#plt.scatter([i for i in range(10)],[res[i][1][1][0] for i in range(10)],color='r',marker = '.')

plt.show()
