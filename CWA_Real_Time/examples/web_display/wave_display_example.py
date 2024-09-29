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


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def connect_earthworm():
    socketio.emit('connect_init')


def get_wave():
    while True:
        if earthworm.mod_sta() is False:
            continue
        time.sleep(0.001)

        wave = earthworm.get_wave(0)
        if wave:
            if "Z" not in wave["channel"]:
                continue

            # GDMS TSMIP new format to old format
            if wave['network'] == 'TW':
                wave['network'] = 'SM'
                wave['location'] = '01'

            trace_name = f'{wave["station"]}.{wave["channel"]}.{wave["network"]}.{wave["location"]}'

            socketio.emit(
                'earthquake_data', {
                    'station': trace_name,
                    'endt': int(wave['endt'] * 1000),
                    'data': wave['data'].tolist()
                }
            )


def get_pick():
    while True:
        pick_msg = earthworm.get_msg(buf_ring=2, msg_type=0)
        if pick_msg:
            try:
                pick_info = pick_msg.split()
                station = pick_info[0]
                channel = pick_info[1]
                network = pick_info[2]
                location = pick_info[3]
                lon = pick_info[4]
                lat = pick_info[5]
                pa = pick_info[6]
                pv = pick_info[7]
                pd = pick_info[8]
                pick_time = pick_info[10]
                weight = pick_info[11]
                repeat = pick_info[13]


                print(f'{station}.{channel}.{network}.{location} '
                      f'{pick_time} {weight} {repeat}')
                socketio.emit('pick_data', {
                    'station': f'{station}.{channel}.{network}.{location}',
                    'longitude': lon,
                    'latitude': lat,
                    'pga': pa,
                    'pgv': pv,
                    'pd': pd,
                    'pick_time': pick_time,
                    'weight': weight,
                    'repeat': repeat
                })

            except IndexError:
                continue
        time.sleep(0.001)


if __name__ == '__main__':
    earthworm = PyEW.EWModule(def_ring=1000, mod_id=2, inst_id=255,
                              hb_time=30, db=False)
    earthworm.add_ring(1000)
    earthworm.add_ring(1002)
    earthworm.add_ring(1005)

    threading.Thread(target=get_wave, daemon=True).start()
    threading.Thread(target=get_pick, daemon=True).start()
    app.run(host='192.168.100.238', port=5000)
    socketio.run(app, debug=True)
