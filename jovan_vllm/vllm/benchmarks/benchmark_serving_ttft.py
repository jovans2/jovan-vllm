"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    (vLLM backend)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_hf_server.sh <your_model>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --tokenizer <your_model> --dataset <target_dataset> \
        --request-rate <request_rate>
"""
import json
import time
from typing import List, Tuple

import requests
import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer

# (prompt len, output len, latency)
REQUEST_LATENCY = []

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

    len_10 = "", 0, 0
    len_50 = "", 0, 0
    len_100 = "", 0, 0
    len_200 = "", 0, 0
    len_500 = "", 0, 0
    len_800 = "", 0, 0
    len_1000 = "", 0, 0
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if 5 < prompt_len < 15 and len_10[0] == "":
            len_10 = prompt, prompt_len, output_len
        if 40 < prompt_len < 60 and len_50[0] == "":
            len_50 = prompt, prompt_len, output_len
        if 90 < prompt_len < 110 and len_100[0] == "":
            len_100 = prompt, prompt_len, output_len
        if 190 < prompt_len < 210 and len_200[0] == "":
            len_200 = prompt, prompt_len, output_len
        if 490 < prompt_len < 510 and len_500[0] == "":
            len_500 = prompt, prompt_len, output_len
        if 790 < prompt_len < 810 and len_800[0] == "":
            len_800 = prompt, prompt_len, output_len
        if 990 < prompt_len < 1010 and len_1000[0] == "":
            len_1000 = prompt, prompt_len, output_len

    filtered_dataset: List[Tuple[str, int, int]] = [len_10, len_50, len_100, len_200, len_500, len_800, len_1000]
    return filtered_dataset


async def send_request(backend: str, model: str, api_url: str, prompt: str,
                       prompt_len: int, output_len: int, best_of: int,
                       use_beam_search: bool, pbar: tqdm) -> None:
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

    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))
    pbar.update(1)

def main():
    global REQUEST_LATENCY
    tokenizer = get_tokenizer("meta-llama/Llama-2-70b-hf", trust_remote_code=False)
    dataset = sample_requests("../../../../ShareGPT_V3_unfiltered_cleaned_split.json", tokenizer)

    api_url = f"http://localhost:8000/generate"

    for data in dataset:
        pbar = tqdm(total=1000)
        for _ in range(1000):
            send_request("vllm", None, api_url, data[0], data[1], data[2], 1, False, pbar)
        pbar.close()
        latencies = []
        for lat in REQUEST_LATENCY:
            latencies.append(lat[0])
        print("Average latency = ", sum(latencies)/len(latencies))
        print("P50 latency = ", np.percentile(latencies, 50))
        print("P99 latency = ", np.percentile(latencies, 99))
        REQUEST_LATENCY = []


if __name__ == "__main__":
    main()
