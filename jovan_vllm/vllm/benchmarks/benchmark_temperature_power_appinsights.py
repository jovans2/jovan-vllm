import json
import ast
import sys
import time
import re
from typing import List, Tuple
import os
import threading
import requests
import numpy as np
import subprocess
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer

from opencensus.stats import aggregation as aggregation_module
from opencensus.stats import measure as measure_module
from opencensus.stats import stats as stats_module
from opencensus.stats import view as view_module
from opencensus.tags import tag_map as tag_map_module
from opencensus.ext.azure import metrics_exporter

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

m_power_w0 = measure_module.MeasureFloat("repl/ttft", "TTFT Latency", "ms")
stats = stats_module.stats
view_manager0 = stats.view_manager
stats_recorder0 = stats.stats_recorder
mmap0 = stats_recorder0.new_measurement_map()
tmap0 = tag_map_module.TagMap()
power_view0 = view_module.View(f"ttft_{my_az_name}",
                               "The TTFT latency measurements",
                               [],
                               m_power_w0,
                               aggregation_module.LastValueAggregation())
view_manager0.register_view(power_view0)
exporter0 = metrics_exporter.new_metrics_exporter(connection_string=
                                                  os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager0.register_exporter(exporter0)


RANDOM_SEED = 100
OVERSAMPLING_FACTOR = 2


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

    mmap0.measure_float_put(m_power_w0, ttft_number)
    mmap0.record(tmap0)
    
    pattern = r"MY TBT = (\d+\.\d+)"

    match = re.search(pattern, rsp)
    tbt_number = float(match.group(1))

    ttfts.append(ttft_number)
    tbts.append(tbt_number)
    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))
    # pbar.update(1)


def main(load_reqs, reqt):
    global REQUEST_LATENCY
    global ttfts
    global tbts
    tokenizer = get_tokenizer("meta-llama/Meta-Llama-3-70b-Instruct", trust_remote_code=False)
    dataset = sample_requests("../../../../ShareGPT_V3_unfiltered_cleaned_split.json", tokenizer)
    data = dataset[reqt]
    print(data[1], "-", data[2])
    api_url = f"http://localhost:8000/generate"
    
    instance_events = generate_poisson_distribution([load_reqs], 600)[0]
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

    latencies = []
    print("START LATENCIES")
    for lat in REQUEST_LATENCY:
        print(lat)
        latencies.append(lat[2])
    print("END LATENCIES")

    print("Average ttft = ", sum(ttfts)/len(ttfts))
    print("P50 ttft = ", np.percentile(ttfts, 50))
    print("P99 ttft = ", np.percentile(ttfts, 99))
    print("Average tbt = ", sum(tbts)/len(tbts))
    print("P50 tbt = ", np.percentile(tbts, 50))
    print("P99 tbt = ", np.percentile(tbts, 99))
    
    REQUEST_LATENCY = []

    return np.percentile(ttfts, 99)


if __name__ == "__main__":

    reqts = [2, 6]
    loads = {2: [3.5, 8.0, 16], 6: [0.5, 1.5, 2.5]}
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
                ttft = main(load, reqt)

                time.sleep(600)
            
