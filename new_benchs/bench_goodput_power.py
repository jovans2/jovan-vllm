# USAGE: python3 bench_goodput.py --MED_IN=1024 --MED_OUT=350 --REQ_LOW=4 --REQ_STEP=1 --SLO_TTFT=100 --SLO_TPOT=50 --model=meta-llama/Llama-3.1-8B 

import subprocess
import sys
import numpy as np
import re
import csv
import argparse
import time
import requests
import threading
import signal
import statistics
import os
from itertools import product

def wait_for_server(timeout=1200):
    url = "http://localhost:8000"
    for _ in range(timeout):
        try:
            r = requests.get(url)
            if r.status_code in [200, 404]:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def start_server(tp_size, model):
    print(f"Starting vLLM server with TP={tp_size}")
    return subprocess.Popen([
        "vllm", "serve", str(model),
        "--swap-space", "16",
        "--disable-log-requests",
        "--tensor-parallel-size", str(tp_size),
        "--gpu-memory-utilization", "0.95",
    ])

def stop_server(proc):
    print(f"Killing server PID {proc.pid}")
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()

def start_process_dcgmi():
    command = "dcgmi dmon -d 100 -e 140,150,157,100,101 > dcgm_monitor_results"
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

def parse_output(output):
    stats = {}
    for line in output.splitlines():
        if match := re.match(r"P99 TTFT \(ms\):\s+([\d.]+)", line):
            stats['P99_TTFT'] = float(match.group(1))
        elif match := re.match(r"Mean TTFT \(ms\):\s+([\d.]+)", line):
            stats['P50_TTFT'] = float(match.group(1))
        elif match := re.match(r"P99 TPOT \(ms\):\s+([\d.]+)", line):
            stats['P99_TPOT'] = float(match.group(1))
        elif match := re.match(r"Mean TPOT \(ms\):\s+([\d.]+)", line):
            stats['P50_TPOT'] = float(match.group(1))
    return stats

def run_benchmark(input_len, output_len, req_rate, model):
    cmd = [
        "python3", "benchmark_serving.py",
        "--backend", "vllm",
        "--dataset-name", "random",
        "--request-rate", str(req_rate),
        "--num-prompts", str(int(20*req_rate)),
        "--ignore-eos",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--model", model
    ]
    try:
        result = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return parse_output(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark: {e.stderr}")
        return None

def get_dmon_line_count():
    if not os.path.exists("dcgm_monitor_results"):
        return 0
    with open("dcgm_monitor_results", "r") as f:
        return len(f.readlines())

def read_dmon_data_between(start_line_idx):
    if not os.path.exists("dcgm_monitor_results"):
        return {}

    with open("dcgm_monitor_results", "r") as f:
        lines = f.readlines()

    new_lines = lines[start_line_idx:]
    data_lines = [line for line in new_lines if line.startswith("GPU ")]

    columns = {'POWINST': [], 'MTMPTR': [], 'TMPTR': [], 'SMCLK': [], 'MMCLK': []}
    for line in data_lines:
        parts = line.strip().split()
        try:
            columns['MTMPTR'].append(float(parts[2]))
            columns['TMPTR'].append(float(parts[3]))
            columns['POWINST'].append(float(parts[4]))
            columns['SMCLK'].append(float(parts[5]))
            columns['MMCLK'].append(float(parts[6]))
        except Exception:
            continue

    result = {}
    for k, v in columns.items():
        if v:
            result[f'P50_{k}'] = round(statistics.median(v), 2)
            result[f'P99_{k}'] = round(np.percentile(v, 99), 2)
        else:
            result[f'P50_{k}'] = None
            result[f'P99_{k}'] = None

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--MED_IN', type=int, required=True)
    parser.add_argument('--MED_OUT', type=int, required=True)
    parser.add_argument('--REQ_LOW', type=float, required=True)
    parser.add_argument('--REQ_STEP', type=float, required=True)
    parser.add_argument('--SLO_TTFT', type=float, required=True)
    parser.add_argument('--SLO_TPOT', type=float, required=True)
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--output_csv', type=str, default='mm_benchmark_results.csv')
    args = parser.parse_args()

    input_len = args.MED_IN
    output_len = args.MED_OUT
    tp_sizes = [8]
    model = args.model

    thread_dcgmi = threading.Thread(target=check_dcgmi)
    thread_dcgmi.daemon = True
    thread_dcgmi.start()

    all_server_procs = []

    with open(args.output_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "InputLen", "OutputLen", "RequestRate", "TensorParallelism",
            "P50_TTFT", "P99_TTFT", "P50_TPOT", "P99_TPOT",
            "P50_POWINST", "P50_MTMPTR", "P50_TMPTR", "P50_SMCLK", "P50_MMCLK",
            "P99_POWINST", "P99_MTMPTR", "P99_TMPTR", "P99_SMCLK", "P99_MMCLK"
        ])

        for tp in tp_sizes:
            server_proc = start_server(tp, model)
            all_server_procs.append(server_proc)

            if not wait_for_server():
                print("Server failed to start.")
                stop_server(server_proc)
                continue
            print("Server started")

            req_rate = args.REQ_LOW
            while True:
                print(f"TP={tp} | IN={input_len}, OUT={output_len}, RATE={req_rate}")
                time.sleep(20)

                dmon_start_idx = get_dmon_line_count()
                result = run_benchmark(input_len, output_len, req_rate, args.model)
                if not result:
                    break
                dcgm_stats = read_dmon_data_between(dmon_start_idx)
                result.update(dcgm_stats)

                writer.writerow([
                    input_len, output_len, req_rate, tp,
                    result.get("P50_TTFT"), result.get("P99_TTFT"),
                    result.get("P50_TPOT"), result.get("P99_TPOT"),
                    result.get("P50_POWINST"), result.get("P50_MTMPTR"), result.get("P50_TMPTR"),
                    result.get("P50_SMCLK"), result.get("P50_MMCLK"),
                    result.get("P99_POWINST"), result.get("P99_MTMPTR"), result.get("P99_TMPTR"),
                    result.get("P99_SMCLK"), result.get("P99_MMCLK")
                ])
                f.flush()

                print(result)

                if result["P99_TTFT"] > args.SLO_TTFT or result["P99_TPOT"] > args.SLO_TPOT:
                    print(f"SLO violated at rate {req_rate}, confirming with re-run...")
                    time.sleep(20)
                    dmon_start_idx = get_dmon_line_count()
                    confirm_result = run_benchmark(input_len, output_len, req_rate, args.model)
                    if not confirm_result:
                        break
                    confirm_dcgm_stats = read_dmon_data_between(dmon_start_idx)
                    confirm_result.update(confirm_dcgm_stats)

                    writer.writerow([
                        input_len, output_len, req_rate, tp,
                        confirm_result.get("P50_TTFT"), confirm_result.get("P99_TTFT"),
                        confirm_result.get("P50_TPOT"), confirm_result.get("P99_TPOT"),
                        confirm_result.get("P50_POWINST"), confirm_result.get("P50_MTMPTR"), confirm_result.get("P50_TMPTR"),
                        confirm_result.get("P50_SMCLK"), confirm_result.get("P50_MMCLK"),
                        confirm_result.get("P99_POWINST"), confirm_result.get("P99_MTMPTR"), confirm_result.get("P99_TMPTR"),
                        confirm_result.get("P99_SMCLK"), confirm_result.get("P99_MMCLK")
                    ])
                    f.flush()

                    print("Confirm run:", confirm_result)

                    if confirm_result["P99_TTFT"] > args.SLO_TTFT or confirm_result["P99_TPOT"] > args.SLO_TPOT:
                        print(f"SLO violated again at rate {req_rate}, stopping.")
                        break
                    else:
                        print("SLO not violated in confirm run, continuing.")
                        req_rate += args.REQ_STEP
                        continue

                req_rate += args.REQ_STEP

            stop_server(server_proc)

    for proc in all_server_procs:
        if proc.poll() is None:
            stop_server(proc)

if __name__ == "__main__":
    main()

