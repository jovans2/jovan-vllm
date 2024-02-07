import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, time

readFile = open("nvidiasmi_monitor_1000MHz", "r")
readLines = readFile.readlines()
readFile.close()

freqs = []
powers = []
x_points = []
startTime = 0
for line in readLines:
    linelist = line.split(",")
    if linelist[0] == "0":
        freqs.append(float(linelist[5]))
        powers.append(float(linelist[2]))

        time_string = linelist[1].split(" ")[2].split(".")[0]
        datetime_obj = datetime.strptime(time_string, "%H:%M:%S")
        time_obj = datetime_obj.time()
        total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

        if startTime == 0:
            startTime = total_seconds
        pass_time = total_seconds - startTime

        x_points.append(pass_time)

print(min(freqs))

plt.plot(x_points, powers)
plt.show()

