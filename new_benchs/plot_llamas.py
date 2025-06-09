import pandas as pd
import matplotlib.pyplot as plt

# Mapping of filenames to model names for plotting
files = {
    'llama1b_h100.csv': 'LLaMA 1B',
    'llama3b_h100.csv': 'LLaMA 3B',
    'llama8b_h100.csv': 'LLaMA 8B',
    'llama70b_h100.csv': 'LLaMA 70B',
}

# Initialize TTFT and TBT plot data
ttft_data = {}
tbt_data = {}

# Read data
for file, model in files.items():
    df = pd.read_csv(file)
    rps = df['RequestRate']
    ttft = df['P99_TTFT']
    tbt = df['P99_TPOT']
    ttft_data[model] = (rps, ttft)
    tbt_data[model] = (rps, tbt)

# Plot P99 TTFT
plt.figure(figsize=(10, 6))
for model, (rps, ttft) in ttft_data.items():
    plt.plot(rps, ttft, label=model, marker='o')
plt.ylim(0, 400)  # Cut y-axis to 400ms
plt.xlabel('Request Rate (RPS)')
plt.ylabel('P99 TTFT (ms)')
plt.title('P99 TTFT vs Request Rate (H100)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('llama_ttft_h100.png')
plt.close()

# Plot P99 TBT
plt.figure(figsize=(10, 6))
for model, (rps, tbt) in tbt_data.items():
    plt.plot(rps, tbt, label=model, marker='o')
plt.xlabel('Request Rate (RPS)')
plt.ylabel('P99 TBT (ms)')
plt.title('P99 TBT vs Request Rate (H100)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('llama_tbt_h100.png')
plt.close()

