import os
import time

freq = 1980
t1 = time.time()
os.system("sudo nvidia-smi -lgc " + str(freq) + " > /dev/null 2>&1") 
t2 = time.time()
os.system("sudo nvidia-smi -rgc > /dev/null 2>&1")
t3 = time.time()

print(t2-t1)
print(t3-t2)
