from flask import Flask, request, jsonify
import requests
import sys
import os
import time
import threading
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

app = Flask(__name__)
DATA_LOCK = threading.Lock()
ADMITED_TOKENS = 0
CURRENT_LOAD = 0
FREQ_FREQ_S = 5  # Time in seconds to recalculate frequency
MY_TYPE = "LL"
MY_PARALLEL = sys.argv[3]
MY_GPUs = sys.argv[4]
TYPES = ["SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL"]

SLOs = [0.25, 0.5, 1.5]

data_train = pd.read_csv('characterization_energy.csv')
X = data_train[['load', 'parallelization_strategy', 'frequency', 'request_type']]
y = data_train['energy']
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_energy = RandomForestRegressor(n_estimators=100, random_state=42)
model_energy.fit(X_train, y_train)
y_pred = model_energy.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print("Model's (energy) mean absolute percentage error (MAPE) = ", mape)

data_train = pd.read_csv('characterization_performance.csv')
X = data_train[['load', 'parallelization_strategy', 'frequency', 'request_type']]
y = data_train['performance']
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_perf = RandomForestRegressor(n_estimators=100, random_state=42)
model_perf.fit(X_train, y_train)
y_pred = model_perf.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print("Model's (performance) mean absolute percentage error (MAPE) = ", mape)

df = pd.read_csv("AzureLLMInferenceTrace_conv.csv")
df['Timestamp'] = pd.to_datetime(df['TIMESTAMP'])
earliest_timestamp = df['Timestamp'].min()
df['Timestamp'] = (df['Timestamp'] - earliest_timestamp).dt.total_seconds()
timestamps = df['Timestamp'].tolist()
for ind, ts in enumerate(timestamps):
    timestamps[ind] = int(ts)
ts_original = timestamps.copy()
max_ts = max(timestamps)

inputs = df['ContextTokens'].tolist()
outputs = df['GeneratedTokens'].tolist()

short_in = np.percentile(inputs, 33)
medium_in = np.percentile(inputs, 67)
long_in = np.percentile(inputs, 99)

INS = [short_in, medium_in, long_in]

if MY_TYPE[0] == "S":
    MY_INPUT_LEN = short_in
elif MY_TYPE[0] == "M":
    MY_INPUT_LEN = medium_in
else:
    MY_INPUT_LEN = long_in

short_out = np.percentile(outputs, 33)
medium_out = np.percentile(outputs, 67)
long_out = np.percentile(outputs, 99)

OUTS = [short_out, medium_out, long_out]


@app.route('/generate', methods=['POST'])
def process_request():
    global ADMITED_TOKENS
    global MY_TYPE

    data = request.get_json()
    prompt = data.get("prompt", "")
    MY_TYPE = data.get("MY_TYPE", "LL")
    input_len = len(prompt)

    api_url = "http://localhost:" + str(int(sys.argv[1]) + 1000) + "/generate"

    DATA_LOCK.acquire()
    ADMITED_TOKENS += input_len
    DATA_LOCK.release()

    response = requests.post(api_url, json=data)

    return jsonify(response.json()), response.status_code


def calc_load():
    global ADMITED_TOKENS
    global CURRENT_LOAD
    global MY_TYPE

    last_tokens = 0
    frequencies = [800, 1000, 1200, 1400, 1600, 1800, 1980]
    while True:
        time.sleep(FREQ_FREQ_S)
        MY_SLO = SLOs[TYPES.index(MY_TYPE) // 3]
        token_difference = ADMITED_TOKENS - last_tokens
        load = token_difference / FREQ_FREQ_S
        load = load / MY_INPUT_LEN
        energies = []
        performances = []
        for freq in frequencies:
            energies.append(model_energy_func(freq, load, MY_TYPE))
            performances.append(model_perf_func(freq, load, MY_TYPE))
        good_energies = []
        good_freqs = []
        for indP, perf in enumerate(performances):
            if perf <= MY_SLO:
                good_freqs.append(frequencies[indP])
                good_energies.append(energies[indP])

        correct_freq = 1980
        if len(good_energies) > 0:
            min_energy = min(good_energies)
            correct_freq = good_freqs[good_energies.index(min_energy)]
        if load == 0:
            correct_freq = 800
        os.system("sudo nvidia-smi -i " + MY_GPUs + " -lgc " + str(correct_freq) + " > /dev/null 2>&1")
        if correct_freq == 1980:
            os.system("sudo nvidia-smi -i " + MY_GPUs + " -rgc > /dev/null 2>&1")


def model_energy_func(freq, load, reqt):
    dataPoint = {"load": load, "parallelization_strategy": MY_PARALLEL, "frequency": freq,
                 "request_type": reqt}
    dataPoint_df = pd.DataFrame(dataPoint, index=[0])
    dataPoint_encoded = pd.get_dummies(dataPoint_df)
    dataPoint_encoded = dataPoint_encoded.reindex(columns=X.columns, fill_value=0)
    return model_energy.predict(dataPoint_encoded)


def model_perf_func(freq, load, reqt):
    dataPoint = {"load": load, "parallelization_strategy": MY_PARALLEL, "frequency": freq,
                 "request_type": reqt}
    dataPoint_df = pd.DataFrame(dataPoint, index=[0])
    dataPoint_encoded = pd.get_dummies(dataPoint_df)
    dataPoint_encoded = dataPoint_encoded.reindex(columns=X.columns, fill_value=0)
    return model_perf.predict(dataPoint_encoded)


if __name__ == '__main__':
    thread_calc_load = threading.Thread(target=calc_load)
    thread_calc_load.start()

    app.run(debug=True, port=int(sys.argv[2]), host=sys.argv[1])

