import ast
import math
from flask import Flask, request, jsonify
import requests
import threading
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

MY_TYPE = sys.argv[2]
TYPES = ["SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL"]

data_train = pd.read_csv('characterization_energy.csv')
X = data_train[['load', 'parallelization_strategy', 'frequency', 'request_type']]
y = data_train['energy']
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print("Model's mean absolute percentage error (MAPE) = ", mape)

shardsFile = open("shards.txt", "r")
shards_lines = shardsFile.readlines()
MY_SHARDS = shards_lines[TYPES.index(MY_TYPE)].split(": ")[1]
MY_SHARDS = ast.literal_eval(MY_SHARDS)
shardsFile.close()

address_mapping = open("ip_addresses.txt", "r")
address_lines = address_mapping.readlines()
ADDRESSES = {}
for line in address_lines:
    name = line.split(":")[0]
    ip = line.split(": ")[1][:-1]
    ADDRESSES[name] = ip
address_mapping.close()

max_load_tp2 = [5.0, 2.5, 1.0, 2.5, 1.0, 0.5, 0.000001, 0.000001, 0.0000001]
max_load_tp4 = [7.0, 5.5, 4.0, 3.0, 2.6, 1.6, 2.0, 1.5, 0.5]
max_load_tp8 = [9.5, 7.0, 5.0, 3.5, 3.5, 3.0, 2.5, 2.0, 1.9]
MAX_THR = {"TP2": max_load_tp2, "TP4": max_load_tp4, "TP8": max_load_tp8}

REFRESH_INTER = 10

INSTANCE_LOADS = []
INSTANCE_ADDRESSES = []
INSTANCE_THROUGHPUT = []
INSTANCE_TYPE = []
INSTANCE_FREQ = []

CURRENT_EPOCH = -1

DATA_LOCK = threading.Lock()

app = Flask(__name__)


@app.route('/epoch_update', methods=['POST'])
def update_epoch():
    global CURRENT_EPOCH

    CURRENT_EPOCH += 1
    update_pool(CURRENT_EPOCH)

    return jsonify({"Response: ": "OK"}), 200


@app.route('/freq_update', methods=['POST'])
def update_frequency():
    global INSTANCE_FREQ

    DATA_LOCK.acquire()

    data = request.get_json()
    freq = int(data.get("freq", 1980))
    ind = int(data.get("ind", 0))
    INSTANCE_FREQ[ind] = freq

    DATA_LOCK.release()

    return jsonify({"Response: ": "OK"}), 200


def update_pool(current_epoch):
    global INSTANCE_TYPE
    global INSTANCE_ADDRESSES
    global INSTANCE_THROUGHPUT
    global INSTANCE_FREQ
    global INSTANCE_LOADS
    global MY_SHARDS
    global MAX_THR

    DATA_LOCK.acquire()

    INSTANCE_FREQ = []
    INSTANCE_LOADS = []
    INSTANCE_TYPE = []
    INSTANCE_ADDRESSES = []
    INSTANCE_THROUGHPUT = []

    instances_epoch = MY_SHARDS[current_epoch]
    for instance in instances_epoch:
        INSTANCE_FREQ.append(1980)
        INSTANCE_LOADS.append(0)
        INSTANCE_TYPE.append(instance.split("_")[0])
        INSTANCE_ADDRESSES.append(ADDRESSES[instance])
        INSTANCE_THROUGHPUT.append(MAX_THR[instance.split("_")[0]][TYPES.index(MY_TYPE)])

    print(INSTANCE_FREQ)
    print(INSTANCE_LOADS)
    print(INSTANCE_TYPE)
    print(INSTANCE_ADDRESSES)
    print(INSTANCE_THROUGHPUT)

    DATA_LOCK.release()


@app.route('/generate', methods=['POST'])
def process_request():
    global INSTANCE_LOADS
    global INSTANCE_FREQ
    global INSTANCE_TYPE
    global INSTANCE_ADDRESSES
    global INSTANCE_THROUGHPUT

    data = request.get_json()
    DATA_LOCK.acquire()
    TEMP_LOADS = []
    for load in INSTANCE_LOADS:
        TEMP_LOADS.append((load + 1) / REFRESH_INTER)
    ENERGIES = []
    for ind, load in enumerate(TEMP_LOADS):
        if load > INSTANCE_THROUGHPUT[ind]:
            ENERGIES.append(math.inf)
        else:
            ENERGIES.append(model_energy(INSTANCE_TYPE[ind], INSTANCE_FREQ[ind], load))
    min_energy = min(ENERGIES)
    correct_instance = ENERGIES.index(min_energy)
    INSTANCE_LOADS[correct_instance] += 1
    address = INSTANCE_ADDRESSES[correct_instance]

    DATA_LOCK.release()

    api_url = "http://" + address + "/generate"
    response = requests.post(api_url, json=data)

    DATA_LOCK.acquire()
    if correct_instance == INSTANCE_ADDRESSES.index(address):
        INSTANCE_LOADS[correct_instance] -= 1
    DATA_LOCK.release()

    return jsonify(response.json()), response.status_code


def model_energy(p_type, freq, load):
    dataPoint = {"load": load, "parallelization_strategy": p_type, "frequency": freq, "request_type": MY_TYPE}
    dataPoint_df = pd.DataFrame(dataPoint, index=[0])
    dataPoint_encoded = pd.get_dummies(dataPoint_df)
    dataPoint_encoded = dataPoint_encoded.reindex(columns=X.columns, fill_value=0)
    return model.predict(dataPoint_encoded)


if __name__ == '__main__':
    app.run(debug=True, port=int(sys.argv[1]))
