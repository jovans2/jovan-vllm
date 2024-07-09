import json
import ast
import time
import re
from typing import List, Tuple
import os
import threading
import requests
import numpy as np
import subprocess
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer

from opencensus.stats import aggregation as aggregation_module
from opencensus.stats import measure as measure_module
from opencensus.stats import stats as stats_module
from opencensus.stats import view as view_module
from opencensus.tags import tag_map as tag_map_module
from opencensus.ext.azure import metrics_exporter

cmd1 = ['python3', '-m', 'vllm.entrypoints.api_server', '--model', 'meta-llama/Llama-2-70b-chat-hf', '--swap-space', '16', '--disable-log-requests', '--tensor-parallel-size=8', '--max-num-seqs=256']
cmd2 = ['python3', '-m', 'vllm.entrypoints.api_server', '--model', 'meta-llama/Llama-2-70b-chat-hf', '--swap-space', '16', '--disable-log-requests', '--tensor-parallel-size=4', '--max-num-seqs=256']
cmd3 = ['python3', '-m', 'vllm.entrypoints.api_server', '--model', 'meta-llama/Llama-2-70b-chat-hf', '--swap-space', '16', '--disable-log-requests', '--tensor-parallel-size=2', '--max-num-seqs=256']
cmd4 = ['python3', '-m', 'vllm.entrypoints.api_server', '--model', 'meta-llama/Llama-2-7b-chat-hf', '--swap-space', '16', '--disable-log-requests', '--tensor-parallel-size=8', '--max-num-seqs=256']
cmd5 = ['python3', '-m', 'vllm.entrypoints.api_server', '--model', 'meta-llama/Llama-2-13b-chat-hf', '--swap-space', '16', '--disable-log-requests', '--tensor-parallel-size=8', '--max-num-seqs=256']
cmd6 = ['python3', '-m', 'vllm.entrypoints.api_server', '--model', 'meta-llama/Llama-2-70b-chat-hf', '--swap-space', '16', '--disable-log-requests', '--tensor-parallel-size=8', '--max-num-seqs=64']
cmd7 = ['python3', '-m', 'vllm.entrypoints.api_server', '--model', 'meta-llama/Llama-2-70b-chat-hf', '--swap-space', '16', '--disable-log-requests', '--tensor-parallel-size=8', '--max-num-seqs=16']
cmd8 = ['python3', '-m', 'vllm.entrypoints.api_server', '--model', 'meta-llama/Llama-2-70b-chat-hf', '--swap-space', '16', '--disable-log-requests', '--tensor-parallel-size=8', '--max-num-seqs=1']
cmd9 = ['python3', '-m', 'vllm.entrypoints.api_server', '--model', 'Llama-2-70b-chat-hf-awq', '--swap-space', '16', '--disable-log-requests', '--tensor-parallel-size=8', '--max-num-seqs=256', '--quantization=awq']

commands = [cmd1, cmd2, cmd3, cmd4, cmd5, cmd6, cmd7, cmd8, cmd9, cmd1, cmd1]


def start_server(command):
    # Start the process in the background
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return process


AZ_METADATA_IP = "169.254.169.254"
AZ_METADATA_ENDPOINT = f"http://{AZ_METADATA_IP}/metadata/instance"
AZ_SCHEDULED_ENDPOINT = f"http://{AZ_METADATA_IP}/metadata/scheduledevents"

RANDOM_SEED = 100
OVERSAMPLING_FACTOR = 2


def get_az_vm_name():
    headers_l = {'Metadata': 'True'}
    query_params_l = {'api-version': '2019-06-01'}
    rsp_l = requests.get(AZ_METADATA_ENDPOINT, headers=headers_l, params=query_params_l).json()
    if "compute" in rsp_l and "name" in rsp_l["compute"]:
        return rsp_l["compute"]["name"]
    return None


my_az_name = get_az_vm_name()

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

m_temperature_c = measure_module.MeasureFloat("repl/temperature", "Temperature of GPUs", "C")
view_manager_temp = stats.view_manager
stats_recorder_temp = stats.stats_recorder
mmap1_temp = stats_recorder_temp.new_measurement_map()
tmap1_temp = tag_map_module.TagMap()

temp_view = view_module.View(f"temp_{my_az_name}",
                             "The temperature measurements",
                             [],
                             m_temperature_c,
                             aggregation_module.LastValueAggregation())
view_manager_temp.register_view(temp_view)
exporter_temp = metrics_exporter.new_metrics_exporter(connection_string=
                                                      os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager_temp.register_exporter(exporter_temp)

m_memperature_c = measure_module.MeasureFloat("repl/memperature", "Temperature of GPU MEMs", "C")
view_manager_memp = stats.view_manager
stats_recorder_memp = stats.stats_recorder
mmap1_memp = stats_recorder_memp.new_measurement_map()
tmap1_memp = tag_map_module.TagMap()
memp_view = view_module.View(f"memp_{my_az_name}",
                             "The memperature measurements",
                             [],
                             m_memperature_c,
                             aggregation_module.LastValueAggregation())
view_manager_memp.register_view(memp_view)
exporter_memp = metrics_exporter.new_metrics_exporter(connection_string=
                                                      os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager_memp.register_exporter(exporter_memp)

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

latency1_ms = measure_module.MeasureFloat("repl/latency1", "The TBT in ms", "ms")
view_manager_latency1 = stats.view_manager
stats_recorder_latency1 = stats.stats_recorder
mmap1_latency1 = stats_recorder_latency1.new_measurement_map()
tmap1_latency1 = tag_map_module.TagMap()
latency1_view = view_module.View(f"latency1_{my_az_name}",
                                 "The distribution of the TBT latency",
                                 [],
                                 latency1_ms,
                                 aggregation_module.LastValueAggregation())
view_manager_latency1.register_view(latency1_view)
exporter_latency1 = metrics_exporter.new_metrics_exporter(connection_string=
                                                          os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager_latency1.register_exporter(exporter_latency1)


def start_process_dcgmi():
    command = "dcgmi dmon -e 100,101,112,156,157,140,150,203,204 -d 1000 > dcgm_monitor_test"
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
        result = subprocess.run(["tail", "-n", "8", readfile], stdout=subprocess.PIPE)
        output_lines = result.stdout.decode('utf-8').strip().splitlines()
        last_line = output_lines[-1]
        try:
            temp = float(last_line.split()[7])
            memp = float(last_line.split()[8])
        except:
            temp = 30
            memp = 30

        power = 0
        for line in output_lines:
            try:
                power += float(line.split()[6])
            except:
                power += 120

        mmap1.measure_float_put(m_power_w, power)
        mmap1.record(tmap1)

        mmap1_temp.measure_float_put(m_temperature_c, temp)
        mmap1_temp.record(tmap1_temp)

        mmap1_memp.measure_float_put(m_memperature_c, memp)
        mmap1_memp.record(tmap1_memp)


def EnforceActivityWindow(start_time, end_time, instance_events):
    ret = []
    events_abs = [0] + instance_events
    # event_times = [sum(events_abs[:i]) for i in range(1, len(events_abs) + 1)]
    event_times = []
    last_value = 0
    for prevInd in range(1, len(events_abs) + 1):
        last_value += events_abs[prevInd - 1]
        event_times.append(last_value)
    event_times = [e for e in event_times if (e > start_time) and (e < end_time)]
    ret = [event_times[0]] + [event_times[i] - event_times[i - 1] for i in range(1, len(event_times))]
    return ret


def generate_poisson_distribution(rates, duration):
    ret = []
    np.random.seed(RANDOM_SEED)
    for rate in rates:
        beta = 1.0 / rate
        inter_arrivals = list(np.random.exponential(scale=beta, size=int(OVERSAMPLING_FACTOR * duration * rate)))
        ret.append(EnforceActivityWindow(0, duration, inter_arrivals))
    return ret


# (prompt len, output len, latency)
REQUEST_LATENCY = []
ttfts = []
tbts = []


def sample_requests(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    file_path = "prompts"  # Update with the path to your text file

    # Read the contents of the file
    with open(file_path, "r") as file:
        text = file.read()

    # Split the text based on the separator
    texts = text.split("*" * 35)  # 35 asterisks for the separator

    # Remove any leading/trailing whitespace from each text
    prompts = [t.strip() for t in texts if t.strip()]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    # completions = [completion for _, completion in dataset]
    # completion_token_ids = tokenizer(completions).input_ids
    print(len(prompt_token_ids))
    tokenized_dataset = []
    for i in range(len(prompts)):
        output_len = 100
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    my_req_ss = "", 0, 0
    my_req_sm = "", 0, 0
    my_req_sl = "", 0, 0
    my_req_ms = "", 0, 0
    my_req_mm = "", 0, 0
    my_req_ml = "", 0, 0
    my_req_ls = "", 0, 0
    my_req_lm = "", 0, 0
    my_req_ll = "", 0, 0
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if 240 < prompt_len < 290:
            my_req_ss = prompt, prompt_len, 96
            my_req_sm = prompt, prompt_len, 256
            my_req_sl = prompt, prompt_len, 1024
        if 1000 < prompt_len < 1100:
            my_req_ms = prompt, prompt_len, 96
            my_req_mm = prompt, prompt_len, 256
            my_req_ml = prompt, prompt_len, 600

        if 8000 < prompt_len < 8200:
            my_req_ls = prompt, prompt_len, 5
            my_req_lm = prompt, prompt_len, 256
            my_req_ll = prompt, prompt_len, 600

    filtered_dataset: List[Tuple[str, int, int]] = [my_req_ss, my_req_sm, my_req_sl, my_req_ms, my_req_mm, my_req_ml, my_req_ls, my_req_lm, my_req_ll]

    with open("prompts", "w") as text_file:
        text_file.write(my_req_ss[0])
        text_file.write("***********************************")
        text_file.write(my_req_ms[0])
        text_file.write("***********************************")
        text_file.write(my_req_ls[0])

    for elem in filtered_dataset:
        print(elem[1], " - ", elem[2])

    return filtered_dataset


def send_request(backend: str, model: str, api_url: str, prompt: str, prompt_len: int, output_len: int, best_of: int, use_beam_search: bool) -> None:
    global ttfts
    global tbts
    request_start_time = time.perf_counter()

    headers = {"User-Agent": "Benchmark Client"}
    if backend == "vllm":
        pload = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": False,
        }
        if model is not None:
            pload["model"] = model
    elif backend == "tgi":
        assert not use_beam_search
        params = {
            "best_of": best_of,
            "max_new_tokens": output_len,
            "do_sample": True,
            }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")

    response = requests.post(api_url, headers=headers, json=pload)
    if response.status_code != 200:
        print("ERROR = ", response.text)

    rsp = ast.literal_eval(response.text)["text"]
    rsp = rsp[0]
    pattern = r"MY TTFT = (\d+\.\d+)"

    match = re.search(pattern, rsp)
    ttft_number = float(match.group(1))

    pattern = r"MY TBT = (\d+\.\d+)"

    match = re.search(pattern, rsp)
    tbt_number = float(match.group(1))

    ttfts.append(ttft_number)
    tbts.append(tbt_number)
    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))

    mmap1_latency.measure_float_put(latency_ms, ttft_number)
    mmap1_latency.record(tmap1_latency)

    mmap1_latency1.measure_float_put(latency1_ms, tbt_number)
    mmap1_latency1.record(tmap1_latency1)


def main(load_reqs, reqt):
    global REQUEST_LATENCY
    global ttfts
    global tbts
    tokenizer = get_tokenizer("meta-llama/Meta-Llama-3-70b-Instruct", trust_remote_code=False)
    dataset = sample_requests("../../../ShareGPT_V3_unfiltered_cleaned_split.json", tokenizer)
    data = dataset[reqt]
    print(data[1], "-", data[2])
    api_url = f"http://localhost:8000/generate"

    instance_events = generate_poisson_distribution([load_reqs], 900)[0]
    after_time, before_time = 0, 0
    st = 0
    threads = []
    for t in instance_events:
        st = st + t - (after_time - before_time)
        before_time = time.time()
        if st > 0:
            time.sleep(st)
        thread = threading.Thread(target=send_request, args=("vllm", None, api_url, data[0], data[1], data[2], 1, False))
        thread.start()
        threads.append(thread)
        after_time = time.time()

    for thread in threads:
        thread.join()


if __name__ == "__main__":

    thread_dcgmi = threading.Thread(target=check_dcgmi)
    thread_dcgmi.start()

    thread_export_metrics = threading.Thread(target=export_metrics)
    thread_export_metrics.start()

    for cmd in commands:
        # Start the server
        server_process = start_server(cmd)

        print("Server started with PID:", server_process.pid)

        # Sleep for 5 minutes
        print("Sleeping for 5 minutes before killing the server...")
        time.sleep(300)  # 300 seconds = 5 minutes

        print("Starting the load...")

        reqts = [4]
        loads = {4: [0.5, 1.5, 2.3]}
        freqs = [1980]

        for reqt in reqts:
            for freq in freqs:
                os.system("sudo nvidia-smi -lgc " + str(freq))
                if freq == 1980:
                    os.system("sudo nvidia-smi -rgc")
                for load in loads[reqt]:
                    ttfts = []
                    tbts = []
                    reqtt = reqt
                    main(load, reqt)
                    time.sleep(600)
