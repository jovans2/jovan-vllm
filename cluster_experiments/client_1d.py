import pandas as pd
import numpy as np
import requests
import threading
import time
import sys
import re
import os
import subprocess

from opencensus.stats import aggregation as aggregation_module
from opencensus.stats import measure as measure_module
from opencensus.stats import stats as stats_module
from opencensus.stats import view as view_module
from opencensus.tags import tag_map as tag_map_module
from opencensus.ext.azure import metrics_exporter

MY_ID = int(sys.argv[1])

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

with open("prompts", "r") as file:
    text = file.read()
texts = text.split("*" * 35)
prompts = [t.strip() for t in texts if t.strip()]

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
exporter = metrics_exporter.new_metrics_exporter(connection_string=
                                                 os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager.register_exporter(exporter)

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
exporter_latency = metrics_exporter.new_metrics_exporter(connection_string=
                                                         os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager_latency.register_exporter(exporter_latency)

df = pd.read_csv('conversation_0514_single_nocache.csv')
df['Timestamp'] = pd.to_datetime(df['PreciseTimeStamp'], format="mixed")
earliest_timestamp = df['Timestamp'].min()
df['Timestamp'] = (df['Timestamp'] - earliest_timestamp).dt.total_seconds()
timestamps = df['Timestamp'].tolist()
for ind, ts in enumerate(timestamps):
    timestamps[ind] = int(ts)
max_ts = max(timestamps)

inputs = df['context'].tolist()
outputs = df['sampling'].tolist()

short_in = np.percentile(inputs, 33)
medium_in = np.percentile(inputs, 67)
long_in = np.percentile(inputs, 99)

INS = [short_in, medium_in, long_in]

short_out = np.percentile(outputs, 33)
medium_out = np.percentile(outputs, 67)
long_out = np.percentile(outputs, 99)

OUTS = [short_out, medium_out, long_out]


def send_req(headers, payload):
    api_url = "http://localhost:8000/generate"
    response = requests.post(api_url, headers=headers, json=payload)

    pattern = r"MY TTFT = (\d+\.\d+)"

    match = re.search(pattern, response.text)
    ttft_number = float(match.group(1)) * 1000.0

    mmap1_latency.measure_float_put(latency_ms, ttft_number)
    mmap1_latency.record(tmap1_latency)

    print("TTFT = ", ttft_number)


def generate_load():
    global inputs

    time.sleep(1)

    last_req = 0
    sending_threads = []
    max_inl = [1, 3, 2, 4, 4, 4, 4, 4, 11]
    good_reqs = 0
    for indReq, inputReq in enumerate(inputs):

        timestamp = timestamps[indReq]
        sleep_time = timestamp - last_req
        time.sleep(sleep_time)
        last_req = timestamp

        in_type = 2
        if inputReq <= INS[0]:
            in_type = 0
        elif inputReq <= INS[1]:
            in_type = 1
        out_type = 2
        if out_type <= OUTS[0]:
            out_type = 0
        elif out_type <= OUTS[1]:
            out_type = 1

        req_type = in_type * 3 + out_type

        if req_type == MY_ID:

            good_reqs += 1

            if good_reqs % max_inl[req_type] == 0:

                correct_input = prompts[in_type]
                correct_len = OUTS[out_type]

                headers = {"User-Agent": "Benchmark Client"}
                payload = {
                    "prompt": correct_input,
                    "n": 1,
                    "best_of": 1,
                    "use_beam_search": False,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "max_tokens": correct_len,
                    "ignore_eos": True,
                    "stream": False,
                }

                thread_send = threading.Thread(target=send_req, args=(headers, payload))
                thread_send.start()

                sending_threads.append(thread_send)

    for thr in sending_threads:
        thr.join()

    return 0


def start_process_dcgmi():
    first_gpu = "0"
    command = "dcgmi dmon -i " + first_gpu + " -e 100,101,112,156,157,140,150,203,204 -d 1000 > dcgm_monitor_test"
    return subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def check_process_dcgmi(process):
    return process.poll() is None


def restart_process_dcgmi(process):
    process.kill()
    return start_process_dcgmi()


def check_dcgmi():
    process = start_process_dcgmi()
    while True:
        time.sleep(20)
        if not check_process_dcgmi(process):
            process = restart_process_dcgmi(process)


def export_metrics():
    readfile = "dcgm_monitor_test"
    while True:
        time.sleep(1)
        result = subprocess.run(["tail", "-n", "1", readfile], stdout=subprocess.PIPE)
        last_line = result.stdout.decode('utf-8').strip()
        try:
            power = float(last_line.split()[6])
        except:
            power = 120.0

        mmap1.measure_float_put(m_power_w, power)
        mmap1.record(tmap1)


if __name__ == '__main__':

    thread_dcgmi = threading.Thread(target=check_dcgmi)
    thread_dcgmi.start()

    thread_export_metrics = threading.Thread(target=export_metrics)
    thread_export_metrics.start()

    thread_load = threading.Thread(target=generate_load)
    thread_load.start()

    thread_load.join()
