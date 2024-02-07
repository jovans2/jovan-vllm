import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, time

readFile = open("nvidiasmi_monitor_diff_freq", "r")
readLines = readFile.readlines()
readFile.close()

# POWER = 2, SM FREQ = 8, MEM FREQ = 10, TEMPERATURE = 11
freqs_sm = []
freqs_mem = []
temps = []
powers = []
x_points = []
startTime = 0
for line in readLines:
    linelist = line.split(",")
    if linelist[0] == "0":
        freqs_sm.append(float(linelist[8]))
        freqs_mem.append(float(linelist[10]))
        powers.append(float(linelist[2]))
        temps.append(float(linelist[11]))

        time_string = linelist[1].split(" ")[2].split(".")[0]
        datetime_obj = datetime.strptime(time_string, "%H:%M:%S")
        time_obj = datetime_obj.time()
        total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

        if startTime == 0:
            startTime = total_seconds
        pass_time = total_seconds - startTime

        x_points.append(pass_time)

plt.plot(x_points, powers)
plt.grid(axis="both")
plt.show()

