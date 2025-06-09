import pandas as pd
import matplotlib.pyplot as plt

# Mapping of filenames to model names
files = {
    'llama1b_h100.csv': 'LLaMA 1B',
    'llama3b_h100.csv': 'LLaMA 3B',
    'llama8b_h100.csv': 'LLaMA 8B',
    'llama70b_h100.csv': 'LLaMA 70B',
}

# Initialize data holders
ttft_data = {}
tbt_data = {}
power_data = {}

# Read and parse data
for file, model in files.items():
    df = pd.read_csv(file)
    rps = df['RequestRate']
    ttft_data[model] = (rps, df['P99_TTFT'])
    tbt_data[model] = (rps, df['P99_TPOT'])
    power_data[model] = (rps, df['P99_POWINST'])

# === Plot: P99 TTFT ===
plt.figure(figsize=(10, 6))
for model, (rps, ttft) in ttft_data.items():
    plt.plot(rps, ttft, label=model, marker='o')
plt.ylim(0, 400)
plt.xlabel('Request Rate (RPS)')
plt.ylabel('P99 TTFT (ms)')
plt.title('P99 TTFT vs Request Rate (H100)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('llama_ttft_h100.png')
plt.close()

# === Plot: P99 TBT ===
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

# === Plot: P99 Power ===
plt.figure(figsize=(10, 6))
for model, (rps, power) in power_data.items():
    plt.plot(rps, power, label=model, marker='o')
plt.xlabel('Request Rate (RPS)')
plt.ylabel('P99 Power (W)')
plt.title('P99 Power vs Request Rate (H100)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('llama_power_h100.png')
plt.close()

