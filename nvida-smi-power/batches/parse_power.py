import sys
import numpy as np

inFile = open(sys.argv[1], "r")
readlines = inFile.readlines()
inFile.close()

powers = []
print(readlines[0])
readlines = readlines[1:]
for line in readlines:
    print(line.split(", "))
    power = float(line.split(", ")[2])
    # powers.append(power)
    if power > 80:
        powers.append(power)

print(min(powers))
print(max(powers))
print(sum(powers)/len(powers))
print(np.percentile(powers, 50))
print(np.percentile(powers, 90))
