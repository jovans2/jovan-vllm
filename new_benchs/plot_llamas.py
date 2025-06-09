import pandas as pd
import matplotlib.pyplot as plt

# Define models and accelerators
model_sizes = ['1b', '3b', '8b', '70b']
accelerators = ['h100', 'a100']
file_template = 'llama{}_{}.csv'

# Legend labels
labels = {
    '1b': 'LLaMA 1B',
    '3b': 'LLaMA 3B',
    '8b': 'LLaMA 8B',
    '70b': 'LLaMA 70B'
}

# Marker style per accelerator
markers = {
    'h100': 'o',   # circles
    'a100': 'D'    # diamonds
}

# Storage dictionaries
ttft_data = {}
tbt_data = {}
power_data = {}

# Load all files
for model in model_sizes:
    for accel in accelerators:
        fname = file_template.format(model, accel)
        label = f"{labels[model]} ({accel.upper()})"
        try:
            df = pd.read_csv(fname)
            rps = df['RequestRate']
            ttft_data[label] = (rps, df['P99_TTFT'], markers[accel])
            tbt_data[label] = (rps, df['P99_TPOT'], markers[accel])
            power_data[label] = (rps, df['P99_POWINST'], markers[accel])
        except FileNotFoundError:
            print(f"Warning: {fname} not found. Skipping.")

# === Plot TTFT ===
plt.figure(figsize=(10, 6))
for label, (rps, ttft, marker) in ttft_data.items():
    plt.plot(rps, ttft, label=label, marker=marker)
plt.ylim(0, 400)
plt.xlabel('Request Rate (RPS)')
plt.ylabel('P99 TTFT (ms)')
plt.title('P99 TTFT vs Request Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('llama_ttft.png')
plt.close()

# === Plot TBT ===
plt.figure(figsize=(10, 6))
for label, (rps, tbt, marker) in tbt_data.items():
    plt.plot(rps, tbt, label=label, marker=marker)
plt.xlabel('Request Rate (RPS)')
plt.ylabel('P99 TBT (ms)')
plt.title('P99 TBT vs Request Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('llama_tbt.png')
plt.close()

# === Plot Power ===
plt.figure(figsize=(10, 6))
for label, (rps, power, marker) in power_data.items():
    plt.plot(rps, power, label=label, marker=marker)
plt.xlabel('Request Rate (RPS)')
plt.ylabel('P99 Power (W)')
plt.title('P99 Power vs Request Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('llama_power.png')
plt.close()

