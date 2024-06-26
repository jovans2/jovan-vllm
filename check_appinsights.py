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

m_power_w0 = measure_module.MeasureFloat("repl/power0", "Power consumption of GPU 0", "W")
stats = stats_module.stats
view_manager0 = stats.view_manager
stats_recorder0 = stats.stats_recorder
mmap0 = stats_recorder0.new_measurement_map()
tmap0 = tag_map_module.TagMap()
power_view0 = view_module.View(f"power_{my_az_name}_0",
                               "The power consumption measurements",
                               [],
                               m_power_w0,
                               aggregation_module.LastValueAggregation())
view_manager0.register_view(power_view0)
exporter0 = metrics_exporter.new_metrics_exporter(connection_string=
                                                  os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager0.register_exporter(exporter0)

m_power_w1 = measure_module.MeasureFloat("repl/power1", "Power consumption of GPU 1", "W")
view_manager1 = stats.view_manager
stats_recorder1 = stats.stats_recorder
mmap1 = stats_recorder1.new_measurement_map()
tmap1 = tag_map_module.TagMap()
power_view1 = view_module.View(f"power_{my_az_name}_1",
                               "The power consumption measurements",
                               [],
                               m_power_w1,
                               aggregation_module.LastValueAggregation())
view_manager1.register_view(power_view1)
exporter1 = metrics_exporter.new_metrics_exporter(connection_string=
                                                  os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager1.register_exporter(exporter1)

m_power_w2 = measure_module.MeasureFloat("repl/power2", "Power consumption of GPU 2", "W")
view_manager2 = stats.view_manager
stats_recorder2 = stats.stats_recorder
mmap2 = stats_recorder2.new_measurement_map()
tmap2 = tag_map_module.TagMap()
power_view2 = view_module.View(f"power_{my_az_name}_2",
                               "The power consumption measurements",
                               [],
                               m_power_w2,
                               aggregation_module.LastValueAggregation())
view_manager2.register_view(power_view2)
exporter2 = metrics_exporter.new_metrics_exporter(connection_string=
                                                  os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager2.register_exporter(exporter2)

m_power_w3 = measure_module.MeasureFloat("repl/power3", "Power consumption of GPU 3", "W")
view_manager3 = stats.view_manager
stats_recorder3 = stats.stats_recorder
mmap3 = stats_recorder3.new_measurement_map()
tmap3 = tag_map_module.TagMap()
power_view3 = view_module.View(f"power_{my_az_name}_3",
                               "The power consumption measurements",
                               [],
                               m_power_w3,
                               aggregation_module.LastValueAggregation())
view_manager3.register_view(power_view3)
exporter3 = metrics_exporter.new_metrics_exporter(connection_string=
                                                  os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager3.register_exporter(exporter3)

m_power_w4 = measure_module.MeasureFloat("repl/power4", "Power consumption of GPU 4", "W")
view_manager4 = stats.view_manager
stats_recorder4 = stats.stats_recorder
mmap4 = stats_recorder4.new_measurement_map()
tmap4 = tag_map_module.TagMap()
power_view4 = view_module.View(f"power_{my_az_name}_4",
                               "The power consumption measurements",
                               [],
                               m_power_w4,
                               aggregation_module.LastValueAggregation())
view_manager4.register_view(power_view4)
exporter4 = metrics_exporter.new_metrics_exporter(connection_string=
                                                  os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager4.register_exporter(exporter4)

m_power_w5 = measure_module.MeasureFloat("repl/power5", "Power consumption of GPU 5", "W")
view_manager5 = stats.view_manager
stats_recorder5 = stats.stats_recorder
mmap5 = stats_recorder5.new_measurement_map()
tmap5 = tag_map_module.TagMap()
power_view5 = view_module.View(f"power_{my_az_name}_5",
                               "The power consumption measurements",
                               [],
                               m_power_w5,
                               aggregation_module.LastValueAggregation())
view_manager5.register_view(power_view5)
exporter5 = metrics_exporter.new_metrics_exporter(connection_string=
                                                  os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager5.register_exporter(exporter5)

m_power_w6 = measure_module.MeasureFloat("repl/power6", "Power consumption of GPU 6", "W")
view_manager6 = stats.view_manager
stats_recorder6 = stats.stats_recorder
mmap6 = stats_recorder6.new_measurement_map()
tmap6 = tag_map_module.TagMap()
power_view6 = view_module.View(f"power_{my_az_name}_6",
                               "The power consumption measurements",
                               [],
                               m_power_w6,
                               aggregation_module.LastValueAggregation())
view_manager6.register_view(power_view6)
exporter6 = metrics_exporter.new_metrics_exporter(connection_string=
                                                  os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager6.register_exporter(exporter6)

m_power_w7 = measure_module.MeasureFloat("repl/power7", "Power consumption of GPU 7", "W")
view_manager7 = stats.view_manager
stats_recorder7 = stats.stats_recorder
mmap7 = stats_recorder7.new_measurement_map()
tmap7 = tag_map_module.TagMap()
power_view7 = view_module.View(f"power_{my_az_name}_7",
                               "The power consumption measurements",
                               [],
                               m_power_w7,
                               aggregation_module.LastValueAggregation())
view_manager7.register_view(power_view7)
exporter7 = metrics_exporter.new_metrics_exporter(connection_string=
                                                  os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager7.register_exporter(exporter7)

m_temp_w0 = measure_module.MeasureFloat("repl/temp0", "temp consumption of GPU 0", "W")
view_manager0t = stats.view_manager
stats_recorder0t = stats.stats_recorder
mmap0t = stats_recorder0t.new_measurement_map()
tmap0t = tag_map_module.TagMap()
temp_view0t = view_module.View(f"temp_{my_az_name}_0",
                               "The temp consumption measurements",
                               [],
                               m_temp_w0,
                               aggregation_module.LastValueAggregation())
view_manager0t.register_view(temp_view0t)
exporter0t = metrics_exporter.new_metrics_exporter(connection_string=
                                                   os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager0t.register_exporter(exporter0t)

m_temp_w1 = measure_module.MeasureFloat("repl/temp1", "temp consumption of GPU 1", "W")
view_manager1t = stats.view_manager
stats_recorder1t = stats.stats_recorder
mmap1t = stats_recorder1t.new_measurement_map()
tmap1t = tag_map_module.TagMap()
temp_view1t = view_module.View(f"temp_{my_az_name}_1",
                               "The temp consumption measurements",
                               [],
                               m_temp_w1,
                               aggregation_module.LastValueAggregation())
view_manager1t.register_view(temp_view1t)
exporter1t = metrics_exporter.new_metrics_exporter(connection_string=
                                                   os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager1t.register_exporter(exporter1t)

m_temp_w2 = measure_module.MeasureFloat("repl/temp2", "temp consumption of GPU 2", "W")
view_manager2t = stats.view_manager
stats_recorder2t = stats.stats_recorder
mmap2t = stats_recorder2t.new_measurement_map()
tmap2t = tag_map_module.TagMap()
temp_view2t = view_module.View(f"temp_{my_az_name}_2",
                               "The temp consumption measurements",
                               [],
                               m_temp_w2,
                               aggregation_module.LastValueAggregation())
view_manager2t.register_view(temp_view2t)
exporter2t = metrics_exporter.new_metrics_exporter(connection_string=
                                                   os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager2t.register_exporter(exporter2t)

m_temp_w3 = measure_module.MeasureFloat("repl/temp3", "temp consumption of GPU 3", "W")
view_manager3t = stats.view_manager
stats_recorder3t = stats.stats_recorder
mmap3t = stats_recorder3t.new_measurement_map()
tmap3t = tag_map_module.TagMap()
temp_view3t = view_module.View(f"temp_{my_az_name}_3",
                               "The temp consumption measurements",
                               [],
                               m_temp_w3,
                               aggregation_module.LastValueAggregation())
view_manager3t.register_view(temp_view3t)
exporter3t = metrics_exporter.new_metrics_exporter(connection_string=
                                                   os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager3t.register_exporter(exporter3t)

m_temp_w4 = measure_module.MeasureFloat("repl/temp4", "temp consumption of GPU 4", "W")
view_manager4t = stats.view_manager
stats_recorder4t = stats.stats_recorder
mmap4t = stats_recorder4t.new_measurement_map()
tmap4t = tag_map_module.TagMap()
temp_view4t = view_module.View(f"temp_{my_az_name}_4",
                               "The temp consumption measurements",
                               [],
                               m_temp_w4,
                               aggregation_module.LastValueAggregation())
view_manager4t.register_view(temp_view4t)
exporter4t = metrics_exporter.new_metrics_exporter(connection_string=
                                                   os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager4t.register_exporter(exporter4t)

m_temp_w5 = measure_module.MeasureFloat("repl/temp5", "temp consumption of GPU 5", "W")
view_manager5t = stats.view_manager
stats_recorder5t = stats.stats_recorder
mmap5t = stats_recorder5t.new_measurement_map()
tmap5t = tag_map_module.TagMap()
temp_view5t = view_module.View(f"temp_{my_az_name}_5",
                               "The temp consumption measurements",
                               [],
                               m_temp_w5,
                               aggregation_module.LastValueAggregation())
view_manager5t.register_view(temp_view5t)
exporter5t = metrics_exporter.new_metrics_exporter(connection_string=
                                                   os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager5t.register_exporter(exporter5t)

m_temp_w6 = measure_module.MeasureFloat("repl/temp6", "temp consumption of GPU 6", "W")
view_manager6t = stats.view_manager
stats_recorder6t = stats.stats_recorder
mmap6t = stats_recorder6t.new_measurement_map()
tmap6t = tag_map_module.TagMap()
temp_view6t = view_module.View(f"temp_{my_az_name}_6",
                               "The temp consumption measurements",
                               [],
                               m_temp_w6,
                               aggregation_module.LastValueAggregation())
view_manager6t.register_view(temp_view6t)
exporter6t = metrics_exporter.new_metrics_exporter(connection_string=
                                                   os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager6t.register_exporter(exporter6t)

m_temp_w7 = measure_module.MeasureFloat("repl/temp7", "temp consumption of GPU 7", "W")
view_manager7t = stats.view_manager
stats_recorder7t = stats.stats_recorder
mmap7t = stats_recorder7t.new_measurement_map()
tmap7t = tag_map_module.TagMap()
temp_view7t = view_module.View(f"temp_{my_az_name}_7",
                               "The temp consumption measurements",
                               [],
                               m_temp_w7,
                               aggregation_module.LastValueAggregation())
view_manager7t.register_view(temp_view7t)
exporter7t = metrics_exporter.new_metrics_exporter(connection_string=
                                                   os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager7t.register_exporter(exporter7t)


m_memp_w0 = measure_module.MeasureFloat("repl/memp0", "memp consumption of GPU 0", "W")
view_manager0m = stats.view_manager
stats_recorder0m = stats.stats_recorder
mmap0m = stats_recorder0m.new_measurement_map()
tmap0m = tag_map_module.TagMap()
memp_view0t = view_module.View(f"memp_{my_az_name}_0",
                               "The memp consumption measurements",
                               [],
                               m_memp_w0,
                               aggregation_module.LastValueAggregation())
view_manager0m.register_view(memp_view0t)
exporter0m = metrics_exporter.new_metrics_exporter(connection_string=
                                                   os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager0m.register_exporter(exporter0m)

m_memp_w1 = measure_module.MeasureFloat("repl/memp1", "memp consumption of GPU 1", "W")
view_manager1m = stats.view_manager
stats_recorder1m = stats.stats_recorder
mmap1m = stats_recorder1m.new_measurement_map()
tmap1m = tag_map_module.TagMap()
memp_view1t = view_module.View(f"memp_{my_az_name}_1",
                               "The memp consumption measurements",
                               [],
                               m_memp_w1,
                               aggregation_module.LastValueAggregation())
view_manager1m.register_view(memp_view1t)
exporter1m = metrics_exporter.new_metrics_exporter(connection_string=
                                                   os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager1m.register_exporter(exporter1m)

m_memp_w2 = measure_module.MeasureFloat("repl/memp2", "memp consumption of GPU 2", "W")
view_manager2m = stats.view_manager
stats_recorder2m = stats.stats_recorder
mmap2m = stats_recorder2m.new_measurement_map()
tmap2m = tag_map_module.TagMap()
memp_view2t = view_module.View(f"memp_{my_az_name}_2",
                               "The memp consumption measurements",
                               [],
                               m_memp_w2,
                               aggregation_module.LastValueAggregation())
view_manager2m.register_view(memp_view2t)
exporter2m = metrics_exporter.new_metrics_exporter(connection_string=
                                                   os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager2m.register_exporter(exporter2m)

m_memp_w3 = measure_module.MeasureFloat("repl/memp3", "memp consumption of GPU 3", "W")
view_manager3m = stats.view_manager
stats_recorder3m = stats.stats_recorder
mmap3m = stats_recorder3m.new_measurement_map()
tmap3m = tag_map_module.TagMap()
memp_view3t = view_module.View(f"memp_{my_az_name}_3",
                               "The memp consumption measurements",
                               [],
                               m_memp_w3,
                               aggregation_module.LastValueAggregation())
view_manager3m.register_view(memp_view3t)
exporter3m = metrics_exporter.new_metrics_exporter(connection_string=
                                                   os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager3m.register_exporter(exporter3m)

m_memp_w4 = measure_module.MeasureFloat("repl/memp4", "memp consumption of GPU 4", "W")
view_manager4m = stats.view_manager
stats_recorder4m = stats.stats_recorder
mmap4m = stats_recorder4m.new_measurement_map()
tmap4m = tag_map_module.TagMap()
memp_view4t = view_module.View(f"memp_{my_az_name}_4",
                               "The memp consumption measurements",
                               [],
                               m_memp_w4,
                               aggregation_module.LastValueAggregation())
view_manager4m.register_view(memp_view4t)
exporter4m = metrics_exporter.new_metrics_exporter(connection_string=
                                                   os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager4m.register_exporter(exporter4m)

m_memp_w5 = measure_module.MeasureFloat("repl/memp5", "memp consumption of GPU 5", "W")
view_manager5m = stats.view_manager
stats_recorder5m = stats.stats_recorder
mmap5m = stats_recorder5m.new_measurement_map()
tmap5m = tag_map_module.TagMap()
memp_view5t = view_module.View(f"memp_{my_az_name}_5",
                               "The memp consumption measurements",
                               [],
                               m_memp_w5,
                               aggregation_module.LastValueAggregation())
view_manager5m.register_view(memp_view5t)
exporter5m = metrics_exporter.new_metrics_exporter(connection_string=
                                                   os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager5m.register_exporter(exporter5m)

m_memp_w6 = measure_module.MeasureFloat("repl/memp6", "memp consumption of GPU 6", "W")
view_manager6m = stats.view_manager
stats_recorder6m = stats.stats_recorder
mmap6m = stats_recorder6m.new_measurement_map()
tmap6m = tag_map_module.TagMap()
temp_view6m = view_module.View(f"memp_{my_az_name}_6",
                               "The memp consumption measurements",
                               [],
                               m_memp_w6,
                               aggregation_module.LastValueAggregation())
view_manager6m.register_view(temp_view6m)
exporter6m = metrics_exporter.new_metrics_exporter(connection_string=
                                                   os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager6t.register_exporter(exporter6m)

m_memp_w7 = measure_module.MeasureFloat("repl/memp7", "memp consumption of GPU 7", "W")
view_manager7m = stats.view_manager
stats_recorder7m = stats.stats_recorder
mmap7m = stats_recorder7m.new_measurement_map()
tmap7m = tag_map_module.TagMap()
memp_view7t = view_module.View(f"memp_{my_az_name}_7",
                               "The memp consumption measurements",
                               [],
                               m_memp_w7,
                               aggregation_module.LastValueAggregation())
view_manager7m.register_view(memp_view7t)
exporter7m = metrics_exporter.new_metrics_exporter(connection_string=
                                                   os.environ['APPLICATIONINSIGHTS_CONNECTION_STRING'])
view_manager7m.register_exporter(exporter7m)


def export_metrics():
    readfile = "dcgm_monitor_test"
    while True:
        time.sleep(1)
        result = subprocess.run(["tail", "-n", "8", readfile], stdout=subprocess.PIPE)
        last_line = result.stdout.decode('utf-8').strip()
        last_lines = last_line.split("\n")
        try:
            power0 = float(last_lines[0].split()[6])
            power1 = float(last_lines[1].split()[6])
            power2 = float(last_lines[2].split()[6])
            power3 = float(last_lines[3].split()[6])
            power4 = float(last_lines[4].split()[6])
            power5 = float(last_lines[5].split()[6])
            power6 = float(last_lines[6].split()[6])
            power7 = float(last_lines[7].split()[6])

            temp0 = float(last_lines[0].split()[8])
            temp1 = float(last_lines[1].split()[8])
            temp2 = float(last_lines[2].split()[8])
            temp3 = float(last_lines[3].split()[8])
            temp4 = float(last_lines[4].split()[8])
            temp5 = float(last_lines[5].split()[8])
            temp6 = float(last_lines[6].split()[8])
            temp7 = float(last_lines[7].split()[8])

            memp0 = float(last_lines[0].split()[7])
            memp1 = float(last_lines[1].split()[7])
            memp2 = float(last_lines[2].split()[7])
            memp3 = float(last_lines[3].split()[7])
            memp4 = float(last_lines[4].split()[7])
            memp5 = float(last_lines[5].split()[7])
            memp6 = float(last_lines[6].split()[7])
            memp7 = float(last_lines[7].split()[7])
        except:
            power0 = 120.0
            power1 = 120.0
            power2 = 120.0
            power3 = 120.0
            power4 = 120.0
            power5 = 120.0
            power6 = 120.0
            power7 = 120.0

            temp0 = 30.0
            temp1 = 30.0
            temp2 = 30.0
            temp3 = 30.0
            temp4 = 30.0
            temp5 = 30.0
            temp6 = 30.0
            temp7 = 30.0

            memp0 = 30.0
            memp1 = 30.0
            memp2 = 30.0
            memp3 = 30.0
            memp4 = 30.0
            memp5 = 30.0
            memp6 = 30.0
            memp7 = 30.0

        mmap0.measure_float_put(m_power_w0, power0)
        mmap0.record(tmap0)

        mmap1.measure_float_put(m_power_w1, power1)
        mmap1.record(tmap1)

        mmap2.measure_float_put(m_power_w2, power2)
        mmap2.record(tmap2)

        mmap3.measure_float_put(m_power_w3, power3)
        mmap3.record(tmap3)

        mmap4.measure_float_put(m_power_w4, power4)
        mmap4.record(tmap4)

        mmap5.measure_float_put(m_power_w5, power5)
        mmap5.record(tmap5)

        mmap6.measure_float_put(m_power_w6, power6)
        mmap6.record(tmap6)

        mmap7.measure_float_put(m_power_w7, power7)
        mmap7.record(tmap7)

        mmap0t.measure_float_put(m_temp_w0, temp0)
        mmap0t.record(tmap0t)

        mmap1t.measure_float_put(m_temp_w1, temp1)
        mmap1t.record(tmap1t)

        mmap2t.measure_float_put(m_temp_w2, temp2)
        mmap2t.record(tmap2t)

        mmap3t.measure_float_put(m_temp_w3, temp3)
        mmap3t.record(tmap3t)

        mmap4t.measure_float_put(m_temp_w4, temp4)
        mmap4t.record(tmap4t)

        mmap5t.measure_float_put(m_temp_w5, temp5)
        mmap5t.record(tmap5t)

        mmap6t.measure_float_put(m_temp_w6, temp6)
        mmap6t.record(tmap6t)

        mmap7t.measure_float_put(m_temp_w7, temp7)
        mmap7t.record(tmap7t)

        mmap0m.measure_float_put(m_memp_w0, memp0)
        mmap0m.record(tmap0m)

        mmap1m.measure_float_put(m_memp_w1, memp1)
        mmap1m.record(tmap1m)

        mmap2m.measure_float_put(m_memp_w2, memp2)
        mmap2m.record(tmap2m)

        mmap3m.measure_float_put(m_memp_w3, memp3)
        mmap3m.record(tmap3m)

        mmap4m.measure_float_put(m_memp_w4, memp4)
        mmap4m.record(tmap4m)

        mmap5m.measure_float_put(m_memp_w5, memp5)
        mmap5m.record(tmap5m)

        mmap6m.measure_float_put(m_memp_w6, memp6)
        mmap6m.record(tmap6m)

        mmap7m.measure_float_put(m_memp_w7, memp7)
        mmap7m.record(tmap7m)


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


if __name__ == '__main__':

    thread_dcgmi = threading.Thread(target=check_dcgmi)
    thread_dcgmi.start()

    thread_exporter = threading.Thread(target=export_metrics)
    thread_exporter.start()
