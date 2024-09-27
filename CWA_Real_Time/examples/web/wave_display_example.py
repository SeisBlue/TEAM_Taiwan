from flask import Flask, render_template
from flask_socketio import SocketIO
import numpy as np
import time
import threading
import random
import PyEW

app = Flask(__name__)
socketio = SocketIO(app)

stations = {}


def generate_station_name():
    return f"Station-{random.randint(1, 999):03d}"


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    global stations
    stations_list = list(stations.keys())
    socketio.emit('init_stations', {'stations': stations_list})


def get_wave():
    wave_ring = PyEW.EWModule(def_ring=1000, mod_id=2, inst_id=255,
                              hb_time=30, db=False)
    wave_ring.add_ring(1000)

    while True:
        if wave_ring.mod_sta() is False:
            continue
        time.sleep(0.001)

        wave = wave_ring.get_wave(0)
        if wave:
            if "Z" not in wave["channel"]:
                continue

            trace_name = (wave["station"]
                          + wave["channel"]
                          + wave["network"]
                          + wave["location"])

            socketio.emit('earthquake_data', {'station': trace_name,
                                              'data': wave['data'].tolist(),})


if __name__ == '__main__':
    threading.Thread(target=get_wave, daemon=True).start()
    app.run(host='192.168.100.238', port=5000)
    socketio.run(app, debug=True)
