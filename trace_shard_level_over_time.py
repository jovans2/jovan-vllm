import numpy as np
from scipy.interpolate import interp1d
from pulp import LpProblem, LpMinimize, LpVariable
from pyomo.environ import *
import sys
import time


def calculate_average_signal(signal_values, energy_values, idle_threshold, count_threshold):
    segments = []
    current_segment = []
    in_idle = False
    lower_values = []
    lower_energy = []

    for ind_e, value in enumerate(signal_values):
        if value < idle_threshold:
            lower_values.append(value)
            lower_energy.append(energy_values[ind_e])
            if not in_idle and len(lower_values) > count_threshold:
                in_idle = True
        else:
            if len(lower_values) < count_threshold:
                for val in lower_energy:
                    current_segment.append(val)
            lower_values = []
            lower_energy = []
            if in_idle:
                if len(current_segment) > 0:
                    segments.append(current_segment)  # End of an idle segment
                    current_segment = []
                in_idle = False
            current_segment.append(energy_values[ind_e])  # Add value to the current
    segments.append(current_segment)
    return segments


max_loads = [9, 7, 5, 3.5, 3.5, 3, 2, 2, 1.5]  # USED FOR TP8
max_loads_1 = [7, 5, 4, 3, 3, 2.5, 2, 2, 1.5]  # USED FOR TP2 and TP4

readFile = open("jovan-vllm/dcgm_monitor_tp2_withsleep_a", "r")
readLines_a = readFile.readlines()
readFile.close()
readFile = open("jovan-vllm/dcgm_monitor_tp2_withsleep_b", "r")
readLines_b = readFile.readlines()
readFile.close()
readFile = open("jovan-vllm/dcgm_monitor_tp2_withsleep_c", "r")
readLines_c = readFile.readlines()
readFile.close()
readFile = open("jovan-vllm/dcgm_monitor_tp2_withsleep_d", "r")
readLines_d = readFile.readlines()
readFile.close()

lines = []
for line in readLines_a:
    lines.append(line)
for line in readLines_b:
    lines.append(line)
for line in readLines_c:
    lines.append(line)
for line in readLines_d:
    lines.append(line)

powers = []
energies = []
timestamps = []
start = -1
last_energy = 60209772611
for ind, line in enumerate(lines):
    ind_gpu = line.split()[1]
    if ind_gpu != "0":
        continue
    try:
        energy = (float(line.split()[5]) - last_energy) / 1000
        last_energy = float(line.split()[5])
        power = float(line.split()[6])
        if power > 150 and start == -1:
            start = ind
        if start == -1:
            continue
        powers.append(power)
        energies.append(energy)
        timestamps.append(((ind - start) // 8) * 0.1)
    except:
        pass

segments_power_tp2 = calculate_average_signal(powers, energies, 140, 80)
for ind, segment in enumerate(segments_power_tp2):
    segments_power_tp2[ind] = (sum(segment) / len(segment)) * 2

readFile = open("jovan-vllm/dcgm_monitor_tp4_withsleep_a", "r")
readLines_a = readFile.readlines()
readFile.close()
readFile = open("jovan-vllm/dcgm_monitor_tp4_withsleep_b", "r")
readLines_b = readFile.readlines()
readFile.close()
readFile = open("jovan-vllm/dcgm_monitor_tp4_withsleep_c", "r")
readLines_c = readFile.readlines()
readFile.close()

lines = []
for line in readLines_a:
    lines.append(line)
for line in readLines_b:
    lines.append(line)
for line in readLines_c:
    lines.append(line)

powers = []
energies = []
timestamps = []
start = -1
last_energy = 26601940046  # TP4
for ind, line in enumerate(lines):
    ind_gpu = line.split()[1]
    if ind_gpu != "0":
        continue
    try:
        energy = (float(line.split()[5]) - last_energy) / 1000
        last_energy = float(line.split()[5])
        power = float(line.split()[6])
        if power > 150 and start == -1:
            start = ind
        if start == -1:
            continue
        powers.append(power)
        energies.append(energy)
        timestamps.append(((ind - start) // 8) * 0.1)
    except:
        pass

segments_power_tp4 = calculate_average_signal(powers, energies, 140, 80)
for ind, segment in enumerate(segments_power_tp4):
    segments_power_tp4[ind] = (sum(segment) / len(segment)) * 4

readFile = open("jovan-vllm/dcgm_monitor_tp8_withsleep_a", "r")
readLines_a = readFile.readlines()
readFile.close()
readFile = open("jovan-vllm/dcgm_monitor_tp8_withsleep_b", "r")
readLines_b = readFile.readlines()
readFile.close()

lines = []
for line in readLines_a:
    lines.append(line)
for line in readLines_b:
    lines.append(line)

powers = []
energies = []
timestamps = []
start = -1
last_energy = 30821804090
for ind, line in enumerate(lines):
    ind_gpu = line.split()[1]
    if ind_gpu != "0":
        continue
    try:
        energy = (float(line.split()[5]) - last_energy) / 1000
        last_energy = float(line.split()[5])
        power = float(line.split()[6])
        if power > 150 and start == -1:
            start = ind
        if start == -1:
            continue
        powers.append(power)
        energies.append(energy)
        timestamps.append(((ind - start) // 8) * 0.1)
    except:
        pass

segments_power_tp8 = calculate_average_signal(powers, energies, 140, 40)
for ind, segment in enumerate(segments_power_tp8):
    segments_power_tp8[ind] = (sum(segment) / len(segment)) * 8

energies_tp8 = []
energies_tp4 = []
energies_tp2 = []

index = 0
for reqt in range(9):
    for freq in range(7):
        for load in range(int(max_loads[reqt] * 2)):
            index += 1
            if freq == 6:
                energies_tp8.append(segments_power_tp8[index - 1])
index = 0
for reqt in range(9):
    for freq in range(7):
        for load in range(int(max_loads_1[reqt] * 2)):
            index += 1
            if freq == 6:
                energies_tp4.append(segments_power_tp4[index - 1])

index = 0
for reqt in range(9):
    for freq in range(7):
        for load in range(int(max_loads_1[reqt] * 2)):
            index += 1
            if freq == 6:
                energies_tp2.append(segments_power_tp2[index - 1])


def compute_optimal_shard(N, TL, RQT):
    t1 = time.time()
    max_load_tp2 = [5.0, 2.5, 1.0, 2.5, 1.0, 0.5, 0.0, 0.0, 0.0]
    max_load_tp4 = [7.0, 5.5, 4.0, 3.0, 2.6, 1.6, 2.0, 1.5, 0.5]
    max_load_tp8 = [9.5, 7.0, 5.0, 3.5, 3.5, 3.0, 2.5, 2.0, 1.9]
    
    max1 = max_load_tp2[RQT]
    max2 = max_load_tp4[RQT]
    max3 = max_load_tp8[RQT]
    
    max_load_tp2 = [7, 5, 4, 3, 3, 2.5, 2, 2, 1.5]
    max_load_tp4 = [7, 5, 4, 3, 3, 2.5, 2, 2, 1.5]
    max_load_tp8 = [9, 7, 5, 3.5, 3.5, 3, 2, 2, 1.5]
    
    load_values1 = [0]
    ml_1 = max_load_tp2[RQT]
    val_l = 0.5
    while val_l <= ml_1:
        load_values1.append(val_l)
        val_l += 0.5
    load_values2 = [0]
    ml_1 = max_load_tp4[RQT]
    val_l = 0.5
    while val_l <= ml_1:
        load_values2.append(val_l)
        val_l += 0.5
    load_values3 = [0]
    ml_1 = max_load_tp8[RQT]
    val_l = 0.5
    while val_l <= ml_1:
        load_values3.append(val_l)
        val_l += 0.5
    
    E1_values = []
    prev_values = 0
    for indR in range(RQT):
        prev_values += max_load_tp2[indR] * 2
    prev_values = int(prev_values)
    for indR in range(int(2 * max_load_tp2[RQT])):
        if indR == 0:
            E1_values.append(energies_tp2[prev_values + indR])
        E1_values.append(energies_tp2[prev_values + indR])
    
    E2_values = []
    prev_values = 0
    for indR in range(RQT):
        prev_values += max_load_tp4[indR] * 2
    prev_values = int(prev_values)
    for indR in range(int(2 * max_load_tp4[RQT])):
        if indR == 0:
            E2_values.append(energies_tp4[prev_values + indR])
        E2_values.append(energies_tp4[prev_values + indR])
    
    E3_values = []
    prev_values = 0
    for indR in range(RQT):
        prev_values += max_load_tp8[indR] * 2
    prev_values = int(prev_values)
    for indR in range(int(2 * max_load_tp8[RQT])):
        if indR == 0:
            E3_values.append(energies_tp8[prev_values + indR])
        E3_values.append(energies_tp8[prev_values + indR])
    
    def e1_func(num):
        a, b = np.polyfit(load_values1, E1_values, 1)
        return a * num + b
    
    
    def e2_func(num):
        a, b = np.polyfit(load_values2, E2_values, 1)
        return a * num + b
    
    
    def e3_func(num):
        a, b = np.polyfit(load_values3, E3_values, 1)
        return a * num + b
    
    # Create a ConcreteModel
    model = ConcreteModel()
    
    # Define decision variables
    model.N1 = Var(within=NonNegativeIntegers)
    model.N2 = Var(within=NonNegativeIntegers)
    model.N3 = Var(within=NonNegativeIntegers)
    model.L1 = Var(within=NonNegativeReals)
    model.L2 = Var(within=NonNegativeReals)
    model.L3 = Var(within=NonNegativeReals)
    
    # Define objective function
    model.obj = Objective(expr=model.N1 * e1_func(model.L1) + model.N2 * e2_func(model.L2) + model.N3 * e3_func(model.L3), sense=minimize)
    
    # Define constraints
    model.const1 = Constraint(expr=1/4 * model.N1 + 1/2 * model.N2 + model.N3 <= N)
    model.const2 = Constraint(expr=model.L1 * model.N1 + model.L2 * model.N2 + model.L3 * model.N3 == TL)
    model.const3 = Constraint(expr=model.L1 <= max1)
    model.const4 = Constraint(expr=model.L2 <= max2)
    model.const5 = Constraint(expr=model.L3 <= max3)
    
    # Nonlinear constraint bounds
    model.const6 = Constraint(expr=model.L1 >= 0)
    model.const7 = Constraint(expr=model.L2 >= 0)
    model.const8 = Constraint(expr=model.L3 >= 0)
    
    # Solve the optimization problem
    solver = SolverFactory('mindtpy')
    results = solver.solve(model, mip_solver='glpk', nlp_solver='ipopt')
    t2 = time.time()
    # Print the optimal values of decision variables
    print("Optimal values:")
    print("N1:", model.N1.value)
    print("N2:", model.N2.value)
    print("N3:", model.N3.value)
    print("Time = ", t2-t1)

Ns = [6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4]
TLs = [8.035248672139064, 6.705987445678416, 6.269000482858523, 4.835248672139063, 6.296764847899565, 5.127522935779816, 3.4509174311926603, 3.31542732979237, 5.8266537904394005, 3.1982617093191688, 6.866006760019314, 6.137952679864799]
RQT = 8

Ns = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
TLs = [2.2, 2.2, 2.2, 2.2, 4.4, 4.4, 4.4, 4.4, 5.6, 6.6, 3.6, 4.0]
RQT = 7

Ns = [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1]
TLs = [3, 4.8, 4.8, 3, 4.8, 4.8, 4.8, 3, 4, 4, 3, 2]
RQT = 6

Ns = [3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2]
TLs = [8.372587917042381, 8.2636789900811541, 7.004688908926962, 8.372587917042381, 5.908385933273219, 5.976284941388638, 5.431109107303877, 4.431109107303877, 5.3404869251578, 5.7404869251578, 5.7404869251578, 5.2622182146077545]
RQT = 5


Ns = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
TLs = [5.0, 5.2, 4.0, 5.0, 5.2, 5.2, 3.5, 3.5, 7.0, 6.0, 4.0, 5.0]
RQT = 4


Ns = [3, 3, 4, 4]
TLs = [1, 1, 1, 1]
RQT = 3


Ns = [1, 1, 1, 1, 1, 1, 1, 1]
TLs = [2.0999999999999996, 0.09999999999999964, 1.0999999999999996, 2.0999999999999996, 4.1, 5.1, 6.1, 6.1, 3.0999999999999996, 3.0999999999999996, 3.0999999999999996, 7.0]
RQT = 2

Ns = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
TLs = [8.0, 7.0, 8.0, 9.0, 10.0, 11.0, 11.0, 11.0, 10.0, 9.0, 9.0, 6.0]
RQT = 1

for ind, n in enumerate(Ns):
    compute_optimal_shard(n, TLs[ind], RQT)


