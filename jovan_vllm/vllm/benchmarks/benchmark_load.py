import json
import ast
import time
import re
from typing import List, Tuple
import os
import threading
import requests
import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer


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

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    my_req = "", 0, 0
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if 1000 < prompt_len < 1100:
            my_req = prompt, prompt_len, 128

    filtered_dataset: List[Tuple[str, int, int]] = [my_req]
    return filtered_dataset


def send_request(backend: str, model: str, api_url: str, prompt: str, prompt_len: int, output_len: int, best_of: int, use_beam_search: bool) -> None:
    global ttfts
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
    ttfts.append(ttft_number)
    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))
    # pbar.update(1)


def main(load_reqs):
    global REQUEST_LATENCY
    global ttfts
    tokenizer = get_tokenizer("meta-llama/Llama-2-70b-hf", trust_remote_code=False)
    dataset = sample_requests("../../../../ShareGPT_V3_unfiltered_cleaned_split.json", tokenizer)
    data = dataset[0]
    print(data[1], "-", data[2])
    api_url = f"http://localhost:8000/generate"

    instance_events = generate_poisson_distribution([load_reqs], 30)[0]
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
    for lat in REQUEST_LATENCY:
        latencies.append(lat[2])
    print("Average latency = ", sum(latencies) / len(latencies))
    print("P50 latency = ", np.percentile(latencies, 50))
    print("P99 latency = ", np.percentile(latencies, 99))
    print("Average ttft = ", sum(ttfts)/len(ttfts))
    print("P50 ttft = ", np.percentile(ttfts, 50))
    print("P99 ttft = ", np.percentile(ttfts, 99))

    REQUEST_LATENCY = []
    ttfts = []


if __name__ == "__main__":
    loads = [0.5, 1.5, 3]
    freqs = [800, 1000, 1200, 1400, 1600, 1800, 1980]
    loads = [3]
    freqs = [1980]
    for freq in freqs:
        os.system("sudo nvidia-smi -lgc " + str(freq))
        if freq == 1980:
            os.system("sudo nvidia-smi -rgc")
        for load in loads:
            main(load)

