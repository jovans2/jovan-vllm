import pandas as pd
import matplotlib.pyplot as plt
import itertools

# Define models and accelerators
model_sizes = ['1b', '3b', '8b', '70b']
accelerators = ['h100', 'a100', 'v100']
file_template = 'llama{}_{}.csv'

# Human-readable labels
model_labels = {
    '1b': 'LLaMA 1B',
    '3b': 'LLaMA 3B',
    '8b': 'LLaMA 8B',
    '70b': 'LLaMA 70B'
}

# Marker per accelerator
marker_style = {
    'h100': 'o',  # Circle
    'a100': 'D',  # Diamond
    'v100': 's',  # Square
}

# Assign consistent color per model
color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
model_colors = {model: next(color_cycle) for model in model_sizes}

# Data containers
ttft_data = {}
tbt_data = {}
power_data = {}

# Load data with deduplication
for model in model_sizes:
    for accel in accelerators:
        file_name = file_template.format(model, accel)
        label = f"{model_labels[model]} ({accel.upper()})"
        try:
            df = pd.read_csv(file_name)
            # Deduplicate by keeping only the last row per RPS
            df = df.groupby('RequestRate', as_index=False).last().sort_values('RequestRate')
            rps = df['RequestRate']
            ttft_data[label] = (rps, df['P99_TTFT'], model_colors[model], marker_style[accel])
            tbt_data[label] = (rps, df['P99_TPOT'], model_colors[model], marker_style[accel])
            power_data[label] = (rps, df['P99_POWINST'], model_colors[model], marker_style[accel])
        except FileNotFoundError:
            print(f"Warning: {file_name} not found. Skipping.")

# Plotting helper
def plot_metric(data, ylabel, title, filename, ylim=None):
    plt.figure(figsize=(10, 6))
    for label, (rps, yvals, color, marker) in data.items():
        plt.plot(rps, yvals, label=label, marker=marker, color=color)
    plt.xlabel('Request Rate (RPS)')
    plt.ylabel(ylabel)
    if ylim:
        plt.ylim(*ylim)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Plot all metrics
plot_metric(ttft_data, 'P99 TTFT (ms)', 'P99 TTFT vs Request Rate', 'llama_ttft.png', ylim=(0, 400))
plot_metric(tbt_data, 'P99 TBT (ms)', 'P99 TBT vs Request Rate', 'llama_tbt.png')
plot_metric(power_data, 'P99 Power (W)', 'P99 Power vs Request Rate', 'llama_power.png')

