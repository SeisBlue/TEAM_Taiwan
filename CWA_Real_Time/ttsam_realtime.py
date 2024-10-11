import argparse
import multiprocessing
import time

import numpy as np
import PyEW
import torch
from flask import Flask, render_template
from flask_socketio import SocketIO

from model.ttsam_model import get_full_model

app = Flask(__name__)
socketio = SocketIO(app)

# 共享物件
manager = multiprocessing.Manager()

wave_buffer = manager.dict()
time_buffer = manager.dict()
pick_buffer = manager.dict()
trigger_buffer = manager.dict()

lock = multiprocessing.Lock()

buffer_time = 30  # 設定緩衝區保留時間
sample_rate = 100  # 設定取樣率


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def connect_earthworm():
    socketio.emit("connect_init")


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

                # add new trace to buffer
                if wave_id not in wave_buffer.keys():
                    wave_buffer[wave_id] = np.zeros(sample_rate * buffer_time)
                    time_buffer[wave_id] = np.append(
                        np.linspace(
                            wave["startt"] - 14,
                            wave["startt"],
                            sample_rate * (buffer_time - 1),
                        ),
                        np.linspace(wave["startt"], wave["endt"], sample_rate),
                    )

                wave_buffer[wave_id] = np.roll(wave_buffer[wave_id],
                                               -wave["data"].size)
                wave_buffer[wave_id][-wave["data"].size:] = wave["data"]

                time_buffer[wave_id] = np.roll(time_buffer[wave_id],
                                               -wave["data"].size)
                time_buffer[wave_id][-wave["data"].size:] = np.linspace(
                    wave["startt"], wave["endt"], wave["data"].size
                )

            except Exception as e:
                print("earthworm_wave_listener error", e)
        time.sleep(0.0001)


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

        pick["traceid"] = join_id_from_dict(pick, order="NSLC")

        return pick

    except IndexError:
        print("pick_msg parsing error:", pick_msg)


def trigger_process(debug=False):
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

        trigger_buffer[pick_id] = {"pick": pick, "trace": trace_dict}

    return trigger_buffer


def converter(trigger_msg, debug=False):
    if debug:
        print("get trigger:", trigger_msg.keys())

    waveform = []
    sta = []
    target = []
    station_name = []
    for i, (pick_id, data) in enumerate(trigger_msg.items()):
        trace = []
        for j, component in enumerate(["Z", "N", "E"]):
            trace.append(data["trace"]["data"][component.lower()])

        waveform.append(trace)
        sta.append(
            [float(data["pick"]["lat"]), float(data["pick"]["lon"]), 100, 760])
        target.append(
            [float(data["pick"]["lat"]), float(data["pick"]["lon"]), 100, 760]
        )

        station_name.append(data["pick"]["station"])

    output = {
        "waveform": waveform,
        "sta": sta,
        "target": target,
        "station_name": station_name,
    }
    return output


def reorder_array(data):
    wave = np.array(data["waveform"])
    wave_transposed = wave.transpose(0, 2, 1)
    data_limit = min(len(data["waveform"]), 25)

    waveform = np.zeros((25, 3000, 3))
    station = np.zeros((25, 4))
    target = np.zeros((25, 4))

    # 取前 25 筆資料，不足的話補 0
    waveform[:data_limit] = wave_transposed[:data_limit]
    station[:data_limit] = data["sta"][:data_limit]
    target[:data_limit] = data["target"][:data_limit]

    input_waveform = torch.tensor(waveform).to(torch.double).unsqueeze(0)
    input_station = torch.tensor(station).to(torch.double).unsqueeze(0)
    target_station = torch.tensor(target).to(torch.double).unsqueeze(0)
    sample = {
        "waveform": input_waveform,
        "sta": input_station,
        "target": target_station,
        "station_name": data["station_name"],
    }
    return sample


def ttsam_model_predict(data, debug=False):
    model_path = f"model/ttsam_trained_model_11.pt"
    full_model = get_full_model(model_path)

    weight, sigma, mu = full_model(data)
    if debug:
        print(f"weight: {weight}")
        print(f"sigma: {sigma}")
        print(f"mu: {mu}")

    return weight, sigma, mu


def trigger_emitter():
    while True:
        if trigger_buffer:
            # 將資料傳送給前端
            socketio.emit("trigger_data", trigger_buffer)
        time.sleep(0.5)


def inference_trigger(debug=False):
    """
    進行模型預測
    """
    while True:
        if pick_buffer:
            try:
                trigger_message = trigger_process()

                output = converter(trigger_message)

                data = reorder_array(output)

                weight, sigma, mu = ttsam_model_predict(data)

            except Exception as e:
                print("trigger_emitter error", e)

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
        inference_trigger,
        trigger_emitter
    ]

    # 為每個函數創建一個持續運行的進程
    for func in functions:
        p = multiprocessing.Process(target=func)
        processes.append(p)
        p.start()

    if args.web or args.host or args.port:
        app.run(host=args.host, port=args.port, use_reloader=False)
        socketio.run(app, debug=True)

    # 主進程要等待這些進程的完成（但由於是服務，不會實際完成）
    for p in processes:
        p.join()
