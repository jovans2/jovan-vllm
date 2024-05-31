from flask import Flask, request, jsonify
import requests
import sys
import re
import os
import time
import threading
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

from opencensus.stats import aggregation as aggregation_module
from opencensus.stats import measure as measure_module
from opencensus.stats import stats as stats_module
from opencensus.stats import view as view_module
from opencensus.tags import tag_map as tag_map_module
from opencensus.ext.azure import metrics_exporter

app = Flask(__name__)
DATA_LOCK = threading.Lock()
ADMITED_TOKENS = 0
CURRENT_LOAD = 0
FREQ_FREQ_S = 5  # Time in seconds to recalculate frequency
MY_TYPE = "LL"
MY_PARALLEL = sys.argv[3]
MY_GPUs = sys.argv[4]
TYPES = ["SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL"]

AZ_METADATA_IP = "169.254.169.254"
AZ_METADATA_ENDPOINT  = f"http://{AZ_METADATA_IP}/metadata/instance"
AZ_SCHEDULED_ENDPOINT = f"http://{AZ_METADATA_IP}/metadata/scheduledevents"


def get_az_vm_name():
    headers_l = {'Metadata': 'True'}
    query_params_l = {'api-version': '2019-06-01'}
    rsp_l = requests.get(AZ_METADATA_ENDPOINT, headers=headers_l, params=query_params_l).json()
    if "compute" in rsp_l and "name" in rsp_l["compute"]:
        return rsp_l["compute"]["name"]
    return None


my_az_name = get_az_vm_name()
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

# Register Power tracker in AppInsights
m_power_w = measure_module.MeasureFloat("repl/power", "Power consumption of GPUs", "W")
stats = stats_module.stats
view_manager = stats.view_manager
stats_recorder = stats.stats_recorder
mmap1 = stats_recorder.new_measurement_map()
tmap1 = tag_map_module.TagMap()
power_view = view_module.View(f"power_{my_az_name}",
                              "The power consumption measurements",
                              [],
                              m_power_w,
                              aggregation_module.LastValueAggregation())
view_manager.register_view(power_view)
exporter = metrics_exporter.new_metrics_exporter(connection_string=os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager.register_exporter(exporter)

# Register Frequency tracker in AppInsights
frequency_mhz = measure_module.MeasureFloat("repl/frequency", "The frequency in MHz of GPUs", "MHz")
view_manager_freq = stats.view_manager
stats_recorder_freq = stats.stats_recorder
mmap1_max_freq = stats_recorder_freq.new_measurement_map()
tmap1_max_freq = tag_map_module.TagMap()
frequency_view = view_module.View(f"frequency_{my_az_name}",
                                  "The distribution of the frequency",
                                  [],
                                  frequency_mhz,
                                  aggregation_module.LastValueAggregation())
view_manager_freq.register_view(frequency_view)
exporter_freq = metrics_exporter.new_metrics_exporter(connection_string=os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager_freq.register_exporter(exporter_freq)

# Register Energy tracker in AppInsights
energy_mj = measure_module.MeasureFloat("repl/energy", "The energy in mJ of GPU cores", "mJ")
view_manager_energy = stats.view_manager
stats_recorder_energy = stats.stats_recorder
mmap1_energy = stats_recorder_energy.new_measurement_map()
tmap1_energy = tag_map_module.TagMap()
energy_view = view_module.View(f"min_frequency_{my_az_name}",
                               "The distribution of the energy",
                               [],
                               energy_mj,
                               aggregation_module.LastValueAggregation())
view_manager_energy.register_view(energy_view)
exporter_energy = metrics_exporter.new_metrics_exporter(connection_string=os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager_energy.register_exporter(exporter_energy)


# Register Energy tracker in AppInsights
latency_ms = measure_module.MeasureFloat("repl/latency", "The TTFT in ms", "ms")
view_manager_latency = stats.view_manager
stats_recorder_latency = stats.stats_recorder
mmap1_latency = stats_recorder_latency.new_measurement_map()
tmap1_latency = tag_map_module.TagMap()
latency_view = view_module.View(f"latency_{my_az_name}",
                                "The distribution of the TTFT latency",
                                [],
                                latency_ms,
                                aggregation_module.LastValueAggregation())
view_manager_latency.register_view(latency_view)
exporter_latency = metrics_exporter.new_metrics_exporter(connection_string=os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager_latency.register_exporter(exporter_latency)


@app.route('/generate', methods=['POST'])
def process_request():
    global ADMITED_TOKENS
    global MY_TYPE

    print("Receive a request!")

    data = request.get_json()
    prompt = data.get("prompt", "")
    MY_TYPE = data.get("MY_TYPE", "LL")
    del data["MY_TYPE"]
    input_len = len(prompt)

    api_url = "http://localhost:" + str(int(sys.argv[2]) + 1000) + "/generate"

    DATA_LOCK.acquire()
    ADMITED_TOKENS += input_len
    DATA_LOCK.release()

    response = requests.post(api_url, json=data)
    pattern = r"MY TTFT = (\d+\.\d+)"

    match = re.search(pattern, response.text)
    ttft_number = float(match.group(1)) * 1000.0

    mmap1_latency.measure_float_put(latency_ms, ttft_number)
    mmap1_latency.record(tmap1_latency)

    print(ttft_number)

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

        print("Next frequency = ", correct_freq)

        mmap1_max_freq.measure_float_put(frequency_mhz, correct_freq)
        mmap1_max_freq.record(tmap1_max_freq)

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


def export_metrics():
    readfile = "dcgm_monitor_test"
    while True:
        time.sleep(5)
        result = subprocess.run(["tail", "-n", "1", readfile], stdout=subprocess.PIPE)
        last_line = result.stdout.decode('utf-8').strip()
        try:
            power = float(last_line.split()[6])
        except:
            power = 300.0

        print("Current power = ", power)

        mmap1.measure_float_put(m_power_w, power)
        mmap1.record(tmap1)


if __name__ == '__main__':

    command = "dcgmi dmon -e 100,101,112,156,157,140,150,203,204 -d 1000 > dcgm_monitor_test"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    thread_calc_load = threading.Thread(target=calc_load)
    thread_calc_load.start()

    thread_export_metrics = threading.Thread(target=export_metrics)
    thread_export_metrics.start()

    app.run(debug=True, port=int(sys.argv[2]), host=sys.argv[1])

