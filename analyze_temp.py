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

tempFile = open("dcgm_monitor_temperature_a", "r")
tempLines = tempFile.readlines()

tempFile = open("dcgm_monitor_temperature_b", "r")
tempLines = tempLines + tempFile.readlines()

tempFile = open("dcgm_monitor_temperature_c", "r")
tempLines = tempLines + tempFile.readlines()

tempFile = open("dcgm_monitor_temperature_d", "r")
tempLines = tempLines + tempFile.readlines()

tempFile = open("dcgm_monitor_temperature_tp4_cold_gpus", "r")
tempLines = tempFile.readlines()

temperatures_mem = []
for _ in range(8):
    temperatures_mem.append([])

temperatures_gpu = []
for _ in range(8):
    temperatures_gpu.append([])

powers_gpu = []
for _ in range(8):
    powers_gpu.append([])

frequencies_gpu = []
for _ in range(8):
    frequencies_gpu.append([])

ind = 0
for line in tempLines:
    try:
        lineList = line.split()
        gpuId = int(lineList[1])
        temp_mem = float(lineList[2])
        temp_gpu = float(lineList[3])
        power_gpu = float(lineList[4])
        freq_gpu = float(lineList[5])

        temperatures_mem[gpuId].append(temp_mem)
        temperatures_gpu[gpuId].append(temp_gpu)
        powers_gpu[gpuId].append(power_gpu)
        frequencies_gpu[gpuId].append(freq_gpu)
    except:
        pass

for gpuId in range(8):
    temperatures_gpu[gpuId] = temperatures_gpu[gpuId][:32000]
    temperatures_mem[gpuId] = temperatures_mem[gpuId][:32000]
    powers_gpu[gpuId] = powers_gpu[gpuId][:32000]
    frequencies_gpu[gpuId] = frequencies_gpu[gpuId][:32000]

x_vals = []
for ind, val in enumerate(temperatures_gpu[0]):
    x_vals.append(ind)
for gpuId in range(len(temperatures_gpu)):
    plt.plot(x_vals, temperatures_gpu[gpuId], label="GPU"+str(gpuId))
plt.xlabel("Time [100ms]")
plt.ylabel("GPU Temp [C]")
plt.grid(axis="both")
plt.legend()
plt.show()
plt.clf()

for gpuId in range(len(temperatures_gpu)):
    plt.plot(x_vals, temperatures_mem[gpuId], label="GPU"+str(gpuId))
plt.xlabel("Time [100ms]")
plt.ylabel("Mem Temp [C]")
plt.grid(axis="both")
plt.legend()
plt.show()
plt.clf()

max_diff_temp = []
for ind in range(len(temperatures_gpu[0])):
    mini_list = []
    for gpuId in range(8):
        mini_list.append(temperatures_gpu[gpuId][ind])
    max_temp = max(mini_list)
    min_temp = min(mini_list)
    max_diff_temp.append(max_temp - min_temp)

plt.plot(x_vals, max_diff_temp)
plt.xlabel("Time [100ms]")
plt.ylabel("Max GPU Temp Diff [C]")
plt.grid(axis="both")
plt.show()

max_diff_temp = []
for ind in range(len(temperatures_mem[0])):
    mini_list = []
    for gpuId in range(8):
        mini_list.append(temperatures_mem[gpuId][ind])
    max_temp = max(mini_list)
    min_temp = min(mini_list)
    max_diff_temp.append(max_temp - min_temp)

plt.plot(x_vals, max_diff_temp)
plt.xlabel("Time [100ms]")
plt.ylabel("Max Mem Temp Diff [C]")
plt.grid(axis="both")
plt.show()
plt.clf()

for gpuId in range(len(temperatures_gpu)):
    plt.plot(x_vals, powers_gpu[gpuId], label="GPU"+str(gpuId))
plt.xlabel("Time [100ms]")
plt.ylabel("GPU Power [W]")
plt.grid(axis="both")
plt.legend()
plt.show()
plt.clf()

max_diff_power = []
for ind in range(len(powers_gpu[0])):
    mini_list = []
    for gpuId in range(8):
        mini_list.append(powers_gpu[gpuId][ind])
    max_temp = max(mini_list)
    min_temp = min(mini_list)
    max_diff_power.append(max_temp - min_temp)

plt.plot(x_vals, max_diff_power)
plt.xlabel("Time [100ms]")
plt.ylabel("Max GPU Power Diff [W]")
plt.grid(axis="both")
plt.show()
plt.clf()

for gpuId in range(len(frequencies_gpu)):
    plt.plot(x_vals, frequencies_gpu[gpuId], label="GPU"+str(gpuId))
plt.xlabel("Time [100ms]")
plt.ylabel("GPU Frequency [MHz]")
plt.grid(axis="both")
plt.legend()
plt.show()
plt.clf()

for gpuId in range(8):
    x = np.sort(powers_gpu[gpuId])
    y = np.arange(len(powers_gpu[gpuId])) / float(len(powers_gpu[gpuId]))
    plt.plot(x, y, label="GPU" + str(gpuId))

plt.grid(axis="both")
plt.xlabel("Power [W]")
plt.ylabel("CDF")
plt.legend()
plt.show()
plt.clf()

for gpuId in range(8):
    x = np.sort(temperatures_gpu[gpuId])
    y = np.arange(len(temperatures_gpu[gpuId])) / float(len(temperatures_gpu[gpuId]))
    plt.plot(x, y, label="GPU" + str(gpuId))

plt.grid(axis="both")
plt.xlabel("Temperature [C]")
plt.ylabel("CDF")
plt.legend()
plt.show()
