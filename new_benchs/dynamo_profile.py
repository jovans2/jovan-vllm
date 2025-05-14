import subprocess
import sys
import re
import csv
import argparse
import time
import requests
import threading
import signal
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
        "--tensor-parallel-size", str(tp_size)
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

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
        "--num-prompts", "100",
        "--ignore-eos",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--model", model
    ]
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return parse_output(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark: {e.stderr}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--SHORT_IN', type=int, required=True)
    parser.add_argument('--MED_IN', type=int, required=True)
    parser.add_argument('--LONG_IN', type=int, required=True)
    parser.add_argument('--SHORT_OUT', type=int, required=True)
    parser.add_argument('--MED_OUT', type=int, required=True)
    parser.add_argument('--LONG_OUT', type=int, required=True)
    parser.add_argument('--REQ_LOW', type=float, required=True)
    parser.add_argument('--REQ_STEP', type=float, required=True)
    parser.add_argument('--SLO_TTFT', type=float, required=True)
    parser.add_argument('--SLO_TPOT', type=float, required=True)
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--output_csv', type=str, default='all_benchmark_results.csv')
    args = parser.parse_args()

    input_lengths = {'S': args.SHORT_IN, 'M': args.MED_IN, 'L': args.LONG_IN}
    output_lengths = {'S': args.SHORT_OUT, 'M': args.MED_OUT, 'L': args.LONG_OUT}
    tp_sizes = [1, 2, 4, 8]
    model = args.model
    configs = list(product("SML", repeat=2))  # SS, SM, ..., LL

    thread_dcgmi = threading.Thread(target=check_dcgmi)
    thread_dcgmi.start()

    with open(args.output_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "InputLen", "OutputLen", "RequestRate", "TensorParallelism",
            "P50_TTFT", "P99_TTFT", "P50_TPOT", "P99_TPOT"
        ])

        for tp in tp_sizes:
            server_proc = start_server(tp, model)
            if not wait_for_server():
                print("Server failed to start.")
                stop_server(server_proc)
                continue

            for in_tag, out_tag in configs:
                input_len = input_lengths[in_tag]
                output_len = output_lengths[out_tag]
                req_rate = args.REQ_LOW

                while True:
                    print(f"TP={tp} | IN={input_len}, OUT={output_len}, RATE={req_rate}")
                    result = run_benchmark(input_len, output_len, req_rate, args.model)
                    if not result:
                        break

                    writer.writerow([
                        input_len, output_len, req_rate, tp,
                        result["P50_TTFT"], result["P99_TTFT"],
                        result["P50_TPOT"], result["P99_TPOT"]
                    ])
                    f.flush()

                    if result["P99_TTFT"] > args.SLO_TTFT or result["P99_TPOT"] > args.SLO_TPOT:
                        print(f"SLO violated at rate {req_rate}")
                        break

                    req_rate += args.REQ_STEP

            stop_server(server_proc)

if __name__ == "__main__":
    main()

