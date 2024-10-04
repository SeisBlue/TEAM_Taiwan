from flask import Flask, render_template
from flask_socketio import SocketIO
import numpy as np
import time
import threading
import PyEW
import argparse
from obspy import UTCDateTime, Trace, Stream
from collections import defaultdict

app = Flask(__name__)
socketio = SocketIO(app)

wave_buffer = {}
time_buffer = {}
picks = {}

buffer_time = 15  # 設定緩衝區保留時間
sample_rate = 100  # 設定取樣率


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def connect_earthworm():
    socketio.emit('connect_init')


def earthworm_wave_listener():
    while True:
        if earthworm.mod_sta() is False:
            continue
        time.sleep(0.00001)

        wave = earthworm.get_wave(0)
        if wave:
            try:
                wave = convert_to_tsmip_legacy_naming(wave)
                wave_id = join_id_from_dict(wave, order='NSLC')

                if wave_id not in wave_buffer.keys():
                    wave_buffer[wave_id] = np.zeros(sample_rate * buffer_time)
                    time_buffer[wave_id] = np.zeros(sample_rate * buffer_time)

                    print(wave_buffer.__len__(), wave_id)
                wave_buffer[wave_id] = np.roll(wave_buffer[wave_id],
                                               -wave['data'].size)
                wave_buffer[wave_id][-wave['data'].size:] = wave['data']


                trace_dict = {'traceid': wave_id,
                              'time': wave['endt'],
                              'data': wave_buffer[wave_id].tolist()}

                if 'Z' in wave['channel']:
                    socketio.emit('trace_data', trace_dict)


            except Exception as e:
                print(e)


def join_id_from_dict(data, order='NSLC'):
    code = {'N': 'network', 'S': 'station', 'L': 'location', 'C': 'channel'}
    data_id = '.'.join(data[code[letter]] for letter in order)
    return data_id


def convert_to_tsmip_legacy_naming(wave):
    if wave['network'] == 'TW':
        wave['network'] = 'SM'
        wave['location'] = '01'

    return wave


def earthworm_pick_listener():
    while True:
        pick_msg = earthworm.get_msg(buf_ring=2, msg_type=0)
        if pick_msg:
            # print(pick_msg)
            try:
                pick_data = parse_pick_msg(pick_msg)
                # socketio.emit('pick_data', pick_data)
            except IndexError:
                continue

        time.sleep(0.001)


def parse_pick_msg(pick_msg):
    pick_msg_column = pick_msg.split()
    try:
        pick = {
            'station': pick_msg_column[0],
            'channel': pick_msg_column[1],
            'network': pick_msg_column[2],
            'location': pick_msg_column[3],
            'lon': pick_msg_column[4],
            'lat': pick_msg_column[5],
            'pga': pick_msg_column[6],
            'pgv': pick_msg_column[7],
            'pd': pick_msg_column[8],
            'tc': pick_msg_column[9],  # Average period
            'pick_time': pick_msg_column[10],
            'weight': pick_msg_column[11],  # 0:best 5:worst
            'instrument': pick_msg_column[12],  # 1:Acc 2:Vel
            'update_sec': pick_msg_column[13]  # sec after pick
        }

        pick['traceid'] = join_id_from_dict(pick, order='NSLC')

        return pick

    except IndexError:
        print('pick_msg parsing error:', pick_msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--web', action='store_true',
                        help='run web server')
    parser.add_argument('--host', type=str, help='web server ip')
    parser.add_argument('--port', type=int, help='web server port')
    args = parser.parse_args()

    earthworm = PyEW.EWModule(def_ring=1000, mod_id=2, inst_id=255,
                              hb_time=30, db=False)
    earthworm.add_ring(1000)  # buf_ring 0: Wave ring(tank player)
    earthworm.add_ring(1002)  # buf_ring 1: Wave ring 2
    earthworm.add_ring(1005)  # buf_ring 2: Pick ring

    try:
        threading.Thread(target=earthworm_wave_listener, daemon=True).start()
        threading.Thread(target=earthworm_pick_listener, daemon=True).start()

        if args.web or args.host or args.port:
            app.run(host=args.host, port=args.port)
            socketio.run(app, debug=True)

    except KeyboardInterrupt:
        earthworm.goodbye()
        print("Exiting...")
