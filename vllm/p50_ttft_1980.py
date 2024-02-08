import numpy as np

fileRead = open("ttft.txt", "r")
lines = fileRead.readlines()
fileRead.close()
lines = lines[-25000:]

ttfs = []
for line in lines:
    if "Time" in line:
        ttfs.append(float(line.split()[-1]))

num_iter = len(ttfs) // 200
print(num_iter)
print(len(ttfs))
#for ind in range(num_iter):
#    print(np.percentile(ttfs[ind * 200: (ind + 1) * 200], 50))
