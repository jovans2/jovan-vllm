import requests
import threading
import time
import pandas as pd
import numpy as np

with open("prompts", "r") as file:
    text = file.read()
texts = text.split("*" * 35)
prompts = [t.strip() for t in texts if t.strip()]

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


def send_req(headers, payload):
    print("Send request")
    api_url = "http://10.0.0.6:" + str(8082) + "/generate"
    requests.post(api_url, headers=headers, json=payload)


def generate_load():
    global inputs

    time.sleep(1)

    inputs = [inputs[0]]

    last_req = 0
    num_good_req = 0
    sending_threads = []
    for ind, input in enumerate(inputs):
        timestamp = timestamps[ind]
        sleep_time = timestamp - last_req
        time.sleep(sleep_time)
        last_req = timestamp

        in_type = 2
        if input <= INS[0]:
            in_type = 0
        elif input <= INS[1]:
            in_type = 1
        out_type = 2
        if out_type <= OUTS[0]:
            out_type = 0
        elif out_type <= OUTS[1]:
            out_type = 1

        if in_type <= 1 and out_type <= 1:

            print("Good request")

            num_good_req += 1
            if num_good_req % 5 == 0:

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
                    "MY_TYPE": "MM",
                }

                thread_send = threading.Thread(target=send_req, args=(headers, payload))
                thread_send.start()

                sending_threads.append(thread_send)

    print("Number of sending threads = ", len(sending_threads))
    for thr in sending_threads:
        thr.join()

    print("I'm done?")

    return 0


if __name__ == '__main__':

    thread_load = threading.Thread(target=generate_load)
    thread_load.start()

    thread_load.join()
