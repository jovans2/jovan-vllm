import matplotlib.pyplot as plt
import numpy as np

# tempFile = open("dcgm_temp_monitor_sleep", "r")
# tempLines = tempFile.readlines()

recoverFile = open("benchmark_temperature.txt", "r")
recoverLines = recoverFile.readlines()
recover_times = []
for line in recoverLines:
    if "All GPUs and memories are cold after" in line:
        recover_times.append(float(line.split()[-1]))

x_vals = []
for ind in range(len(recover_times)):
    x_vals.append(ind)
plt.plot(x_vals, recover_times)
plt.xlabel("Config")
plt.ylabel("Recovery time [s]")
plt.grid(axis="both")
plt.show()
plt.clf()

tempFile = open("dcgm_monitor_temperature_tp4_cold_gpus", "r")
tempLines_cold = tempFile.readlines()

tempFile = open("dcgm_monitor_temperature_tp4_hot_gpus", "r")
tempLines_hot = tempFile.readlines()

tempFile = open("dcgm_monitor_temperature_tp4_all_gpus", "r")
tempLines_all = tempFile.readlines()

temperatures_mem = [[], [], []]
temperatures_gpu = [[], [], []]
powers_gpu = [[], [], []]
frequencies_gpu = [[], [], []]

labels = ["Cold", "Hot", "All"]

ind = 0
for line in tempLines_cold:
    try:
        lineList = line.split()
        gpuId = int(lineList[1])
        temp_mem = float(lineList[2])
        temp_gpu = float(lineList[3])
        power_gpu = float(lineList[4])
        freq_gpu = float(lineList[5])

        if gpuId == 0:
            temperatures_mem[0].append(temp_mem)
            temperatures_gpu[0].append(temp_gpu)
            powers_gpu[0].append(power_gpu)
            frequencies_gpu[0].append(freq_gpu)
    except:
        pass

ind = 0
for line in tempLines_hot:
    try:
        lineList = line.split()
        gpuId = int(lineList[1])
        temp_mem = float(lineList[2])
        temp_gpu = float(lineList[3])
        power_gpu = float(lineList[4])
        freq_gpu = float(lineList[5])

        if gpuId == 0:
            temperatures_mem[1].append(temp_mem)
            temperatures_gpu[1].append(temp_gpu)
            powers_gpu[1].append(power_gpu)
            frequencies_gpu[1].append(freq_gpu)
    except:
        pass

ind = 0
for line in tempLines_all:
    try:
        lineList = line.split()
        gpuId = int(lineList[1])
        temp_mem = float(lineList[2])
        temp_gpu = float(lineList[3])
        power_gpu = float(lineList[4])
        freq_gpu = float(lineList[5])

        if gpuId == 0:
            temperatures_mem[2].append(temp_mem)
            temperatures_gpu[2].append(temp_gpu)
            powers_gpu[2].append(power_gpu)
            frequencies_gpu[2].append(freq_gpu)
    except:
        pass

for gpuId in range(3):
    temperatures_gpu[gpuId] = temperatures_gpu[gpuId][:9000]
    temperatures_mem[gpuId] = temperatures_mem[gpuId][:9000]
    powers_gpu[gpuId] = powers_gpu[gpuId][:9000]
    frequencies_gpu[gpuId] = frequencies_gpu[gpuId][:9000]

x_vals = []
for ind, val in enumerate(temperatures_gpu[0]):
    x_vals.append(ind)
for gpuId in range(len(temperatures_gpu)):
    plt.plot(x_vals, temperatures_gpu[gpuId], label=labels[gpuId])
plt.xlabel("Time [100ms]")
plt.ylabel("GPU Temp [C]")
plt.grid(axis="both")
plt.legend()
plt.show()
plt.clf()

for gpuId in range(len(temperatures_gpu)):
    plt.plot(x_vals, temperatures_mem[gpuId], label=labels[gpuId])
plt.xlabel("Time [100ms]")
plt.ylabel("Mem Temp [C]")
plt.grid(axis="both")
plt.legend()
plt.show()
plt.clf()

for gpuId in range(len(temperatures_gpu)):
    plt.plot(x_vals, powers_gpu[gpuId], label=labels[gpuId])
plt.xlabel("Time [100ms]")
plt.ylabel("GPU Power [W]")
plt.grid(axis="both")
plt.legend()
plt.show()
plt.clf()

for gpuId in range(len(frequencies_gpu)):
    plt.plot(x_vals, frequencies_gpu[gpuId], label=labels[gpuId])
plt.xlabel("Time [100ms]")
plt.ylabel("GPU Frequency [MHz]")
plt.grid(axis="both")
plt.legend()
plt.show()
plt.clf()

for gpuId in range(3):
    x = np.sort(powers_gpu[gpuId])
    y = np.arange(len(powers_gpu[gpuId])) / float(len(powers_gpu[gpuId]))
    plt.plot(x, y, label=labels[gpuId])

plt.grid(axis="both")
plt.xlabel("Power [W]")
plt.ylabel("CDF")
plt.legend()
plt.show()
plt.clf()

for gpuId in range(3):
    x = np.sort(temperatures_gpu[gpuId])
    y = np.arange(len(temperatures_gpu[gpuId])) / float(len(temperatures_gpu[gpuId]))
    plt.plot(x, y, label=labels[gpuId])

plt.grid(axis="both")
plt.xlabel("Temperature [C]")
plt.ylabel("CDF")
plt.legend()
plt.show()
