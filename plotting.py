import numpy as np
import csv

lines = []
with open("results.csv", "r") as file:
    reader = csv.reader(file, delimiter=";")
    for l, row in enumerate(reader) :
        if l != 0:
            # lines.append(row)
            lines.append(
                [int(row[0]), int(row[1]), row[2], 
                float(row[3]), float(row[4]), row[5]])


print(lines)

