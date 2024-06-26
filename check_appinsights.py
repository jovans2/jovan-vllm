import requests
import os
import threading
import subprocess
import time

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

print(my_az_name)


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

if __name__ == '__main__':

    thread_dcgmi = threading.Thread(target=check_dcgmi)
    thread_dcgmi.start()
