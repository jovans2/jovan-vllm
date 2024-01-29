import sys

files = []
batches = [256, 128, 64, 32, 16, 8, 4, 2, 1]
for batch in batches:
    files.append("batch"+str(batch)+".txt")
files = ["beam_search.txt"]
for file in files:
    inFile = open(file, "r")

    readLines = inFile.readlines()

    inFile.close()

    p50s = []
    p99s = []
    for line in readLines:
        if "Jovan --- P50 " in line:
            p50s.append(float(line.split(" ")[-1]))
        elif "Jovan --- P99 " in line:
            p99s.append(float(line.split(" ")[-1]))

    # print(p50s)
    # print(len(p50s))

    # print(p99s)
    # print(len(p99s))

    print(file)
    print(len(p50s))
    for p50 in p50s:
        print(p50)
