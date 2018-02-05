#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 19:47:33 2018

@author: dwu
"""
from __future__ import print_function
import random
import sys
import matplotlib.pyplot as plt
import time
from cop_kmeans import run_cop_kmeans,get_score
from pb_generation import create_pb
from  sklearn.datasets import make_moons, make_blobs
from Deap_GA import call_GA
from PPC import call_CSP
from PLNE import PLNE
import csv
import numpy as np

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def k_means(data,ml,cl,k):
    return 1,[]
def PLNE_g(data,ml,cl,k) :
    return 1,[]
def CSP_yoloj(data,ml,cl,k):
    return 1,[]
def ge(data,ml,cl,k):
    return 1,[]

def is_valid(colors,ml,cl,k):
    #check mustlink
    for x,y in ml:
        if colors[x] != colors[y]:
            return False
    #check mustlink
    for x,y in cl:
        if colors[x] == colors[y]:
            return False
    #check mustlink
    for x in colors:
        if x < 1 or x > k:
            return False
    return True


FORMAT ="{: >10};{: >10};{: >10};{: >10};{: >10};{: >10}"
data = []
ml=[]
cl=[]
k=2
function = [run_cop_kmeans,PLNE,call_CSP,call_GA]



print(FORMAT.format("data_len", "percentage_cons", "function", "time", "score", "truthness"))

def run():
    lendata = len(data)
    lenper = len(percentages)
    lenml = len(ml)
    res = [[{} for i in range(lendata)] for j in range(lenml)]
    for j in range(len(data)):
        test_data=data[j]
        test_ml =ml[j]
        test_cl =cl[j]
        for i in range(len(function)):
            try :
                t,label=function[i](test_data,test_ml,test_cl,k)
                score = get_score(test_data,label,k)
#                 print(t)
                data_len = size_data[j // (lenper)]
                percent = percentages[j % lenper]
                is_true = is_valid(label, test_ml, test_cl, k)
                writer.writerow([data_len, percent, function[i].__name__, t, score, is_true])
                print(FORMAT.format(data_len, percent, function[i].__name__, t, score, is_true))
                res[j // lenper][j % lenper][i]=[t,score]
            except Exception as e:
                eprint("==============================================")
                eprint("data : {} ; cons : {}".format(size_data[j // (lenper)],percentages[j % lenper]))
                eprint(function[i].__name__)
                eprint(e)
                pass
    return res

size_data = [20, 50, 100, 150, 200]
# percentages = [0, 30 , 60]
percentages = np.arange(0, 100, 10)
data_function = make_moons
for a in size_data:
    for c in percentages:
        pb = create_pb(a, (c*a/100,[1,3]), (c*a/100,[1,3]), make_moons, seed=1)
        data.append(pb[0])
        ml.append(pb[1])
        cl.append(pb[2])

with open("results_moons.csv", 'w') as csvfile :
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(["data_len", "percentage_cons", "function", "time", "score", "truthness"])
    res = run()


# size_data = [500, 1000, 2000]
# # percentages = [0, 30 , 60]
# percentages = np.arange(0, 110, 10)
# for a in size_data:
    # for c in percentages:
        # pb = create_pb(a, (c*a/100,[0,2]), (c*a/100,[0,2]), make_blobs, seed=1)
        # data.append(pb[0])
        # ml.append(pb[1])
        # cl.append(pb[2])

# with open("results2.csv", 'w') as csvfile :
    # writer = csv.writer(csvfile, delimiter=';')
    # writer.writerow(["data_len", "percentage_cons", "function", "time", "score", "truthness"])
    # res = run()

plt.scatter([i for i in range(len(data))],[res[i][0][1][0] for i in range(len(data))],color='r',marker = '.')


plt.show()
