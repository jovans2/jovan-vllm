readFile = open("nvidiasmi_monitor_1000MHz", "r")
readLines = readFile.readlines()
readFile.close()

freqs = []
for line in readLines:
    linelist = line.split(",")
    if linelist[0] == "0":
        freqs.append(float(linelist[5]))

print(min(freqs))