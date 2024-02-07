import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, time

readFile = open("dcgm_monitor_check_freqs", "r")
readLines = readFile.readlines()
readFile.close()

# POWER = 2, SM FREQ = 8, MEM FREQ = 10, TEMPERATURE = 11
freqs_sm = []
freqs_mem = []
temps = []
powers = []
x_points = []
startTime = 0
n_curr = 0

power_intervals = []
v_lines_start = []
v_lines_end = []
last_power = 0
for line in readLines:
    linelist = line.split()
    if linelist[1] == "0":
        freqs_sm.append(float(linelist[2]))
        freqs_mem.append(float(linelist[3]))
        powers.append(float(linelist[5]))
        temps.append(float(linelist[4]))

        pass_time = n_curr * 0.1
        n_curr += 1

        power = float(linelist[5])
        if last_power < 150 < power:
            power_intervals.append([])
            v_lines_start.append(pass_time)
        if power < 150 < last_power:
            v_lines_end.append(pass_time)
        if power > 150:
            power_intervals[-1].append(power)
        last_power = power

        x_points.append(pass_time)

plt.plot(x_points, powers)
plt.grid(axis="both")

#for vl in v_lines_start:
#    plt.axvline(vl, color="red")
#for vl in v_lines_end:
#    plt.axvline(vl, color="green")


for interval in power_intervals:
    print(sum(interval)/len(interval))

plt.show()

