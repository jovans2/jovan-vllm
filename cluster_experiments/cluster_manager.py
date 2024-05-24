from flask import Flask, request, jsonify
import requests
import threading
import time
import math
import pandas as pd
import numpy as np

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

short_out = np.percentile(outputs, 33)
medium_out = np.percentile(outputs, 67)
long_out = np.percentile(outputs, 99)

OUTS = [short_out, medium_out, long_out]

POOL_PORTS = ["8081", "8082", "8083", "8084", "8085", "8086", "8087", "8088", "8089"]
POOL_TYPES = ["SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL"]
POOL_AVAILABLE = [1, 1, 1, 1, 1, 1, 1, 1, 1]
POOL_EXIST = [1, 1, 1, 1, 1, 1, 1, 1, 1]
SENT_TOKENS = [0, 0, 0, 0, 0, 0, 0, 0, 0]
RETURNED_TOKENS = [0, 0, 0, 0, 0, 0, 0, 0, 0]
LOAD_PER_TYPE = [0, 0, 0, 0, 0, 0, 0, 0, 0]
DATA_LOCK = threading.Lock()

app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def process_request():
    global POOL_AVAILABLE
    global SENT_TOKENS
    global RETURNED_TOKENS
    global DATA_LOCK
    global LOAD_PER_TYPE

    data = request.get_json()

    prompt = data.get("prompt", "")
    input_len = len(prompt)
    output_len = predict_len(data)
    real_output = data.get("max_tokens", 0)

    in_type = "L"
    out_type = "L"

    if input_len <= INS[0]:
        in_type = "S"
    elif input_len <= INS[1]:
        in_type = "M"

    if output_len <= OUTS[0]:
        out_type = "S"
    elif output_len <= OUTS[1]:
        out_type = "M"

    req_type = in_type + out_type
    pool_ind = POOL_TYPES.index(req_type)

    DATA_LOCK.acquire()

    LOAD_PER_TYPE[pool_ind] += 1

    while POOL_AVAILABLE[pool_ind] == 0:
        in_ind = pool_ind // 3
        out_ind = pool_ind % 3
        if out_ind < 2:
            pool_ind = in_ind * 3 + (out_ind + 1)
        else:
            pool_ind = (in_ind + 1) * 3 + out_ind

    pool_port = POOL_PORTS[pool_ind]

    api_url = "http://localhost:" + str(pool_port) + "/generate"
    SENT_TOKENS[pool_ind] += output_len

    DATA_LOCK.release()

    response = requests.post(api_url, json=data)

    DATA_LOCK.acquire()
    RETURNED_TOKENS[pool_ind] += real_output
    DATA_LOCK.release()

    return jsonify(response.json()), response.status_code


def predict_len(data):
    return data.get("max_tokens", 0)


def avail_pools():
    global SENT_TOKENS
    global RETURNED_TOKENS
    global DATA_LOCK
    while True:
        DATA_LOCK.acquire()
        for pool_ind, pool_exist in enumerate(POOL_EXIST):
            if pool_exist == 1:
                sent_tokens = SENT_TOKENS[pool_ind]
                processed_tokens = RETURNED_TOKENS[pool_ind]
                try:
                    ro = float(sent_tokens / processed_tokens)
                    if ro >= 1:
                        POOL_AVAILABLE[pool_ind] = 0
                    else:
                        POOL_AVAILABLE[pool_ind] = 1
                except ZeroDivisionError:
                    POOL_AVAILABLE[pool_ind] = 1
        DATA_LOCK.release()
        time.sleep(5)


def reset():
    global SENT_TOKENS
    global RETURNED_TOKENS
    global DATA_LOCK
    while True:
        DATA_LOCK.acquire()
        for ind in range(len(SENT_TOKENS)):
            SENT_TOKENS[ind] = 0
            RETURNED_TOKENS[ind] = 0
        DATA_LOCK.release()
        time.sleep(30)


def compute_instances():
    global LOAD_PER_TYPE
    global POOL_EXIST
    global DATA_LOCK

    max_throughput = [9.5, 7.0, 5.0, 3.5, 3.5, 3.0, 2.5, 2.0, 1.9]

    while True:
        DATA_LOCK.acquire()

        for ind in range(len(LOAD_PER_TYPE)):
            in_type = ind // 3
            out_type = ind % 3
            load = LOAD_PER_TYPE[ind]
            if ind != len(LOAD_PER_TYPE) - 1:
                num_inst = load // max_throughput[ind]
            else:
                num_inst = math.ceil(load / max_throughput[ind])

            left_over = load - max_throughput[ind] * num_inst

            if left_over > 0 and ind < 8:
                if out_type < 2:
                    next_type = in_type * 3 + (out_type + 1)
                else:
                    left_over = left_over * INS[in_type] / INS[in_type + 1]
                    next_type = (in_type + 1) * 3 + out_type
                LOAD_PER_TYPE[next_type] += left_over

            if num_inst == 0:
                POOL_EXIST[ind] = 0
            else:
                POOL_EXIST[ind] = 1
            pool_port = POOL_PORTS[ind]
            api_url = "http://localhost:" + str(pool_port) + "/epoch_update"
            data = {'num_instances': num_inst}
            requests.post(api_url, json=data)
            LOAD_PER_TYPE[ind] = 0

        DATA_LOCK.release()

        # EPOCH = 5min
        time.sleep(5 * 60)


if __name__ == '__main__':
    thread_pool_avail = threading.Thread(target=avail_pools)
    thread_pool_avail.start()

    thread_reset = threading.Thread(target=reset)
    thread_reset.start()

    thread_instances = threading.Thread(target=compute_instances)
    thread_instances.start()

    app.run(debug=True, port=8080)
