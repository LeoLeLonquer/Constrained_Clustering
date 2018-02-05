import numpy as np
from PLNE import PLNE
import propagation
from cop_kmeans import run_cop_kmeans
from Deap_GA import call_GA
from PPC import call_CSP
from pb_generation import create_pb
from sklearn.datasets import make_blobs, make_moons
from matplotlib import pyplot as plt

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
# ml_cons=[(1,2),(13,11),(11,22)]
# cl_cons=[(1,7),(13,32),(17,20)]
Klusters = 2
# propagation.propagate(100,ml_cons,cl_cons)

# benchmark.create_pb(100, (0, [0,0]), (0, [0,0]), make_blobs)
data, must_link, cannot_link = create_pb(100, (10, [1,3]), (2, [1,2]), make_moons)

function = run_cop_kmeans
duration, colors = function(data, must_link, cannot_link, Klusters)
print(colors, duration)

ml_groups, cl_groups = propagation.propagate(len(data), must_link , cannot_link)

# print("Validité de la solution: {}".format(
    # is_valid(data, must_link, cannot_link,Klusters)))

# filtered = np_lines[np_lines[:,2]==function]
# filtered = filtered[filtered[:,1]==selected_complexity]
# if function == 'call_GA':
    # filtered_ga = filtered[filtered[:,3].astype(float)< 1500]
    # filtered = filtered_ga
# plt.plot(filtered[:,0].astype(int), 
        # filtered[:,3].astype(float),  
        # dico_color[function], label=name)
# plt.xlabel("Number of points", fontsize=10)
# plt.ylabel("Time(s)", fontsize=10)
# plt.grid(True)
plt.scatter(data[:,0], data[:,1], c=colors)

# Plotting the constraints
# Plotting the equality constraints

for group in ml_groups:
    if len(group) > 1:
        new_X_x = []
        new_X_y = []
        for pt in group :
            new_X_x.append(data[pt, 0])
            new_X_y.append(data[pt, 1])
        plt.plot(new_X_x,new_X_y,
                 '^', linewidth=50)

for label, other_labels in cl_groups.items():
    for o_lab in other_labels:
        pt1 = ml_groups[label].pop() 
        ml_groups[label].add(pt1)
        pt2 = ml_groups[o_lab].pop() 
        ml_groups[o_lab].add(pt2)
        plt.plot([data[pt1,0],data[pt2,0]],
                 [data[pt1,1],data[pt2,1]],
                 'x', linewidth=50)

plt.show()
