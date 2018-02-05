import numpy as np
import csv
from matplotlib import pyplot as plt
lines = []
np_lines = []
with open("results_moons.csv", "r") as file:
    reader = csv.reader(file, delimiter=";")
    for l, row in enumerate(reader) :
        if l != 0:
            # lines.append(row)
            lines.append(
                [int(row[0]), int(row[1]), row[2], 
                float(row[3]), float(row[4]), row[5]])
            np_lines.append(row)
        else :
            fields = row

np_lines = np.array(np_lines)
# print(lines)
print(np_lines)
print(fields)

dico_color = {}
name_function = {}
name_function['call_CSP'] = 'CSP'
name_function['PLNE'] = 'PLNE'
name_function['call_GA'] = 'GA'
name_function['run_cop_kmeans'] = 'Cop-kmeans'
dico_color['call_CSP'] = 'red'
dico_color['PLNE'] = 'blue'
dico_color['call_GA'] = 'green'
dico_color['run_cop_kmeans'] = 'orange'

# print(np_lines[:,2])
map_colors = [dico_color[function] for function in np_lines[:,2]]
map_name = [name_function[function] for function in np_lines[:,2]]

# GRAPH 1 TIME IN FUNCTION OF PB SIZE
# selected_complexity = '40'
# for function, name in name_function.items():
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

# plt.legend()

#GRAPH 2 TIME IN FUNCTION OF NUMBER OF CONSTRAINT
# selected_size = '200'
# for function, name in name_function.items():
    # filtered = np_lines[np_lines[:,2]==function]
    # filtered = filtered[filtered[:,0]==selected_size]
    # if function == 'call_GA':
        # filtered_ga = filtered[filtered[:,3].astype(float)< 1500]
        # filtered = filtered_ga
    # plt.plot(filtered[:,1].astype(int),
            # filtered[:,3].astype(float),
            # dico_color[function], label=name)
    # plt.xlabel("Percentage of connections", fontsize=10)
    # plt.ylabel("Time(s)", fontsize=10)
    # plt.grid(True)

# plt.legend()

# GRAPH 3 SCORE IN FUNCTION OF PB SIZE
# selected_complexity = '40'
# for function, name in name_function.items():
    # filtered = np_lines[np_lines[:,2]==function]
    # filtered = filtered[filtered[:,1]==selected_complexity]
    # if function == 'call_GA':
        # filtered_ga = filtered[filtered[:,3].astype(float)< 1500]
        # filtered = filtered_ga
    # plt.plot(filtered[:,0].astype(int),
            # filtered[:,4].astype(float),
            # dico_color[function], label=name)
    # plt.xlabel("Number of points", fontsize=10)
    # plt.ylabel("Minimized Maximum Diameter", fontsize=10)
    # plt.grid(True)

# plt.legend()

# GRAPH 4 SCORE IN FUNCTION OF NUMBER OF CONSTRAINT
selected_size = '200'
for function, name in name_function.items():
    filtered = np_lines[np_lines[:,2]==function]
    filtered = filtered[filtered[:,0]==selected_size]
    if function == 'call_GA':
        filtered_ga = filtered[filtered[:,3].astype(float)< 1500]
        filtered = filtered_ga
    plt.plot(filtered[:,1].astype(int),
            filtered[:,4].astype(float),
            dico_color[function], label=name)
    plt.xlabel("Percentage of connections", fontsize=10)
    plt.ylabel("Minimized Maximum Diameter", fontsize=10)
    plt.grid(True)

plt.legend()
plt.show()
