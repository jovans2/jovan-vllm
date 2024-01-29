import sys
import numpy as np

inFile = open(sys.argv[1], "r")
readlines = inFile.readlines()[1:]
inFile.close()

powers = []
for line in readlines:
    powers.append(float(line.split(", ")[2]))

print(min(powers))
print(max(powers))
print(sum(powers)/len(powers))
print(np.percentile(powers, 50))
print(np.percentile(powers, 99))
