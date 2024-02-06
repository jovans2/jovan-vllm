import numpy as np

fileRead = open("ttft.txt", "r")
lines = fileRead.readlines()
fileRead.close()

ttfs = []
for line in lines:
    if "Time" in line:
        ttfs.append(float(line.split()[-1]))

num_iter = len(ttfs) // 100
for ind in range(num_iter):
    print(np.percentile(ttfs[ind * 100: (ind + 1) * 100], 50))
