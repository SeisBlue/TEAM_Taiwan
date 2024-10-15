import argparse
import bisect
import multiprocessing
import threading
import time

import numpy as np
import pandas as pd
import PyEW
import torch
from flask import Flask, render_template
from flask_socketio import SocketIO
from scipy.signal import detrend, iirfilter, sosfilt, zpk2sos
from scipy.spatial import cKDTree

from model.ttsam_model import get_full_model

app = Flask(__name__)
socketio = SocketIO(app)

# 共享物件
manager = multiprocessing.Manager()

wave_buffer = manager.dict()
wave_queue = manager.Queue()

time_buffer = manager.dict()
pick_buffer = manager.dict()

event_queue = manager.Queue()




@app.route("/")
def index():
    return render_template("index.html")


@app.route("/trace")
def trace_page():
    return render_template("trace.html")


@app.route("/event")
def event_page():
    return render_template("event.html")


@app.route("/intensityMap")
def map_page():
    return render_template("intensityMap.html")


@socketio.on("connect")
def connect_earthworm():
    socketio.emit("connect_init")


def wave_emitter():
    while True:
        wave = wave_queue.get()
        wave_id = join_id_from_dict(wave, order="NSLC")
        if "Z" not in wave_id:
            continue
        wave["waveid"] = wave_id

        wave_packet = {
            "waveid": wave_id,
            "data": wave["data"].tolist(),
        }
        socketio.emit("wave_packet", wave_packet)


def event_emitter():
    while True:
        event_data = event_queue.get()
        if not event_data:
            continue

        # 將資料傳送給前端
        socketio.emit("event_data", event_data)
        time.sleep(0.5)


def web_server():
    threading.Thread(target=wave_emitter).start()
    threading.Thread(target=event_emitter).start()

    if args.web or args.host or args.port:
        # 開啟 web server
        app.run(host=args.host, port=args.port, use_reloader=False)
        socketio.run(app, debug=True)


def join_id_from_dict(data, order="NSLC"):
    code = {"N": "network", "S": "station", "L": "location", "C": "channel"}
    data_id = ".".join(data[code[letter]] for letter in order)
    return data_id


def convert_to_tsmip_legacy_naming(wave):
    if wave["network"] == "TW":
        wave["network"] = "SM"
        wave["location"] = "01"

    return wave


def earthworm_wave_listener():
    buffer_time = 30  # 設定緩衝區保留時間
    sample_rate = 100  # 設定取樣率
    while True:
        if earthworm.mod_sta() is False:
            continue

        wave = earthworm.get_wave(0)

        if wave:
            """
            這裡並沒有去處理每個 trace 如果時間不連續的問題
            """
            try:
                wave = convert_to_tsmip_legacy_naming(wave)

                wave_id = join_id_from_dict(wave, order="NSLC")

                # 將 wave_id 加入 queue 給 trace_emitter 發送至前端
                if "Z" in wave_id:
                    wave_queue.put(wave)

                # add new trace to buffer
                if wave_id not in wave_buffer.keys():
                    wave_buffer[wave_id] = np.zeros(sample_rate * buffer_time)
                    time_buffer[wave_id] = np.append(
                        np.linspace(
                            wave["startt"] - (buffer_time - 1),
                            wave["startt"],
                            sample_rate * (buffer_time - 1),
                        ),
                        np.linspace(wave["startt"], wave["endt"],
                                    wave["data"].size),
                    )

                wave_buffer[wave_id] = np.append(wave_buffer[wave_id],
                                                 wave["data"])

                wave_buffer[wave_id] = wave_buffer[wave_id][wave["data"].size:]

                time_buffer[wave_id] = np.append(
                    time_buffer[wave_id],
                    np.linspace(wave["startt"], wave["endt"],
                                wave["data"].size),
                )
                time_buffer[wave_id] = time_buffer[wave_id][wave["data"].size:]

            except Exception as e:
                print("earthworm_wave_listener error", e)
        time.sleep(0.000001)


def parse_pick_msg(pick_msg):
    pick_msg_column = pick_msg.split()
    try:
        pick = {
            "station": pick_msg_column[0],
            "channel": pick_msg_column[1],
            "network": pick_msg_column[2],
            "location": pick_msg_column[3],
            "lon": pick_msg_column[4],
            "lat": pick_msg_column[5],
            "pga": pick_msg_column[6],
            "pgv": pick_msg_column[7],
            "pd": pick_msg_column[8],
            "tc": pick_msg_column[9],  # Average period
            "pick_time": pick_msg_column[10],
            "weight": pick_msg_column[11],  # 0:best 5:worst
            "instrument": pick_msg_column[12],  # 1:Acc 2:Vel
            "update_sec": pick_msg_column[13],  # sec after pick
        }

        pick["pickid"] = join_id_from_dict(pick, order="NSLC")

        return pick

    except IndexError:
        print("pick_msg parsing error:", pick_msg)


def earthworm_pick_listener(debug=False):
    """
    監看 pick ring 的訊息，並保存活著的 pick msg
    pick msg 的生命週期為 p 波後 2-9 秒
    ref: pick_ew_new/pick_ra_0709.c line 283
    """
    while True:
        pick_msg = earthworm.get_msg(buf_ring=2, msg_type=0)
        if pick_msg:
            try:
                pick_data = parse_pick_msg(pick_msg)
                pick_id = join_id_from_dict(pick_data, order="NSLC")

                if pick_data["update_sec"] == "2":
                    pick_buffer[pick_id] = pick_data
                    if debug:
                        print("add pick:", pick_id)

                elif pick_data["update_sec"] == "9":
                    pick_buffer.__delitem__(pick_id)
                    if debug:
                        print("delete pick:", pick_id)

            except Exception as e:
                print("earthworm_pick_listener error:", e)
                continue

        time.sleep(0.001)


def get_event(pick_buffer, debug=False):
    event_data = {}
    # pick 只有 Z 軸
    for pick_id, pick in pick_buffer.items():
        network = pick["network"]
        station = pick["station"]
        location = pick["location"]
        channel = pick["channel"]

        data = {}
        # 找到 wave_buffer 內的三軸資料
        for i, component in enumerate(["Z", "N", "E"]):
            wave_id = f"{network}.{station}.{location}.{channel[0:2]}{component}"
            data[component.lower()] = wave_buffer[wave_id].tolist()

        trace_dict = {
            "traceid": pick_id,
            "time": time_buffer[pick_id].tolist(),
            "data": data,
        }

        event_data[pick_id] = {"pick": pick, "trace": trace_dict}

    event_queue.put(event_data)

    return event_data


def signal_processing(wave):
    try:
        # demean and lowpass filter
        data = detrend(wave, type="constant")
        data = lowpass(data, freq=10)
        return data

    except Exception as e:
        print("signal_processing error", e)


def lowpass(data, freq=10, df=100, corners=4):
    """
    Modified form ObsPy Signal Processing
    https://docs.obspy.org/_modules/obspy/signal/filter.html#lowpass
    """
    fe = 0.5 * df
    f = freq / fe

    if f > 1:
        f = 1.0
    z, p, k = iirfilter(corners, f, btype="lowpass", ftype="butter",
                        output="zpk")

    sos = zpk2sos(z, p, k)
    return sosfilt(sos, data)


vs30_table = pd.read_csv(f"data/Vs30ofTaiwan.csv")
tree = cKDTree(vs30_table[["lat", "lon"]])


def get_vs30(pick):
    try:
        lat = float(pick["lat"])
        lon = float(pick["lon"])
        distance, i = tree.query([lat, lon])
        vs30 = vs30_table.iloc[i]["Vs30"]
        return vs30

    except Exception as e:
        print("get_vs30 error", e)


def get_site_info(pick):
    try:
        latitude = float(pick["lat"])
        longitude = float(pick["lon"])
        elevation = 100
        vs30 = get_vs30(pick)
        return [latitude, longitude, elevation, vs30]

    except Exception as e:
        print("get_site_info error", e)


def converter(event_msg, debug=False):
    try:
        if debug:
            print("get trigger:", event_msg.keys())

        waveform = []
        station = []
        target = []
        station_name = []
        for i, (pick_id, data) in enumerate(event_msg.items()):
            trace = []
            for j, component in enumerate(["Z", "N", "E"]):
                wave = data["trace"]["data"][component.lower()]
                wave = signal_processing(wave)
                trace.append(wave)

            waveform.append(trace)
            station.append(get_site_info(data["pick"]))
            target.append(get_site_info(data["pick"]))
            station_name.append(data["pick"]["station"])

        dataset = {
            "waveform": waveform,
            "station": station,
            "target": target,
            "station_name": station_name,
        }
        return dataset
    except Exception as e:
        print("converter error", e)


def reorder_array(data):
    try:
        wave = np.array(data["waveform"])
        wave_transposed = wave.transpose(0, 2, 1)
        data_limit = min(len(data["waveform"]), 25)

        waveform = np.zeros((25, 3000, 3))
        station = np.zeros((25, 4))
        target = np.zeros((25, 4))

        # 取前 25 筆資料，不足的話補 0
        waveform[:data_limit] = wave_transposed[:data_limit]
        station[:data_limit] = data["station"][:data_limit]
        target[:data_limit] = data["target"][:data_limit]

        input_waveform = torch.tensor(waveform).to(torch.double).unsqueeze(0)
        input_station = torch.tensor(station).to(torch.double).unsqueeze(0)
        target_station = torch.tensor(target).to(torch.double).unsqueeze(0)
        sample = {
            "waveform": input_waveform,
            "station": input_station,
            "target": target_station,
            "station_name": data["station_name"],
        }
        return sample
    except Exception as e:
        print("reorder_array error", e)


def ttsam_model_predict(dataset, debug=False):
    model_path = f"model/ttsam_trained_model_11.pt"
    full_model = get_full_model(model_path)
    data = reorder_array(dataset)
    weight, sigma, mu = full_model(data)

    pga_list = torch.sum(weight * mu, dim=2).cpu().detach().numpy().flatten()
    pga_list = pga_list[: len(dataset["station_name"])]

    return pga_list


class TaiwanIntensity:
    label = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
    pga = np.log10(
        [1e-5, 0.008, 0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0]
    )  # log10(m/s^2)
    pgv = np.log10(
        [1e-5, 0.002, 0.007, 0.019, 0.057, 0.15, 0.3, 0.5, 0.8, 1.4]
    )  # log10(m/s)

    def __init__(self):
        self.pga_ticks = self.get_ticks(self.pga)
        self.pgv_ticks = self.get_ticks(self.pgv)

    def calculate(self, pga, pgv=None, label=False):
        pga_intensity = bisect.bisect(self.pga, pga) - 1
        intensity = pga_intensity

        if pga > self.pga[5] and pgv is not None:
            pgv_intensity = bisect.bisect(self.pgv, pgv) - 1
            if pgv_intensity > pga_intensity:
                intensity = pgv_intensity

        if label:
            return self.label[intensity]
        else:
            return intensity

    @staticmethod
    def get_ticks(array):
        ticks = np.cumsum(array, dtype=float)
        ticks[2:] = ticks[2:] - ticks[:-2]
        ticks = ticks[1:] / 2
        ticks = np.append(ticks, (ticks[-1] * 2 - ticks[-2]))
        return ticks


def model_inference(debug=False):
    """
    進行模型預測
    """
    while True:
        if pick_buffer:
            try:
                start_time = time.time()

                event_data = get_event(pick_buffer)
                dataset = converter(event_data)
                pga_list = ttsam_model_predict(dataset)
                intensity = [
                    TaiwanIntensity().calculate(pga, label=True) for pga in
                    pga_list
                ]
                print("intensity:", intensity)

                end_time = time.time()
                print("model_inference time:", end_time - start_time)

            except Exception as e:
                print("inference_trigger error", e)

        time.sleep(0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--web", action="store_true", help="run web server")
    parser.add_argument("--host", type=str, help="web server ip")
    parser.add_argument("--port", type=int, help="web server port")
    args = parser.parse_args()

    earthworm = PyEW.EWModule(
        def_ring=1000, mod_id=2, inst_id=255, hb_time=30, db=False
    )
    earthworm.add_ring(1000)  # buf_ring 0: Wave ring(tank player)
    earthworm.add_ring(1002)  # buf_ring 1: Wave ring 2
    earthworm.add_ring(1005)  # buf_ring 2: Pick ring

    processes = []
    functions = [
        earthworm_wave_listener,
        earthworm_pick_listener,
        model_inference,
        web_server,
    ]

    # 為每個函數創建一個持續運行的進程
    for func in functions:
        p = multiprocessing.Process(target=func)
        processes.append(p)
        p.start()

    # 主進程要等待這些進程的完成（但由於是服務，不會實際完成）
    for p in processes:
        p.join()
