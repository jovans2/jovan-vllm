import numpy as np

file_read = open("vllm/ttft.txt", "r")

ttfts_lines = file_read.readlines()

file_read.close()

low_load = 16
medium_load = 49
high_load = 94
num_freqs = 7
duration = 1

ttfts = []
for line in ttfts_lines:
    ttfts.append(float(line.split(" ")[-1]))
print(len(ttfts))
for freq in range(num_freqs):
    high_ttfts = ttfts[-(high_load * duration):]
    ttfts = ttfts[:len(ttfts)-(high_load * duration)]
    medium_ttfts = ttfts[-(int(medium_load * duration)):]
    ttfts = ttfts[:len(ttfts)-int(medium_load * duration)]
    low_ttfts = ttfts[-(int(low_load * duration)):]

    print(np.percentile(high_ttfts, 50))
    print(np.percentile(medium_ttfts, 50))
    print(np.percentile(low_ttfts, 50))
    # print("********************************")


