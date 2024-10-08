import queue

from flask import Flask, render_template
from flask_socketio import SocketIO
import numpy as np
import time
import threading
import PyEW
import argparse


app = Flask(__name__)
socketio = SocketIO(app)

wave_buffer = {}
time_buffer = {}
picks = {}
trigger_queue = queue.Queue()

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
            """
            這裡並沒有去處理每個 trace 如果時間不連續的問題
            """
            try:
                wave = convert_to_tsmip_legacy_naming(wave)
                wave_id = join_id_from_dict(wave, order='NSLC')

                # add new trace to buffer

                if wave_id not in wave_buffer.keys():
                    wave_buffer[wave_id] = np.zeros(sample_rate * buffer_time)
                    time_buffer[wave_id] = np.append(
                        np.linspace(wave['startt'] - 14, wave['startt'],
                                    sample_rate * (buffer_time - 1)),
                        np.linspace(wave['startt'], wave['endt'], sample_rate))


                wave_buffer[wave_id] = np.roll(wave_buffer[wave_id],
                                               -wave['data'].size)
                wave_buffer[wave_id][-wave['data'].size:] = wave['data']

                time_buffer[wave_id] = np.roll(time_buffer[wave_id],
                                               -wave['data'].size)
                time_buffer[wave_id][-wave['data'].size:] = np.linspace(
                    wave['startt'], wave['endt'], wave['data'].size)

            except Exception as e:
                print(e)


def trigger_emitter(debug=False):
    """
    監看 pick 的更新，並且將波形資料切割成 event
    """

    while True:
        try:
            if debug:
                print('emit wave', picks)

            trigger_msg = {}
            # pick 只有 Z 軸
            for pick_id, pick in picks.items():
                network = pick['network']
                station = pick['station']
                location = pick['location']
                channel = pick['channel']


                data = {}
                # 找到 wave_buffer 內的三軸資料
                for i, component in enumerate(['Z', 'N', 'E']):
                    wave_id = f'{network}.{station}.{location}.{channel[0:2]}{component}'
                    data[component.lower()] = wave_buffer[wave_id].tolist()

                trace_dict = {'traceid': pick_id,
                              'time': time_buffer[pick_id].tolist(),
                              'data': data}

                trigger_msg[pick_id] = {'pick': pick, 'trace': trace_dict}

            # 資料傳給訊號前處理
            trigger_queue.put(trigger_msg)

            # 將資料傳送給前端
            socketio.emit('trigger_data', trigger_msg)

        except Exception as e:
            print(e)
        time.sleep(1)

def earthworm_convert_to_ttsam(debug=True):
    while True:
        trigger_msg = trigger_queue.get()
        if debug:
            print('get trigger:', trigger_msg.keys())
        if trigger_msg:
            pass





def join_id_from_dict(data, order='NSLC'):
    code = {'N': 'network', 'S': 'station', 'L': 'location', 'C': 'channel'}
    data_id = '.'.join(data[code[letter]] for letter in order)
    return data_id


def convert_to_tsmip_legacy_naming(wave):
    if wave['network'] == 'TW':
        wave['network'] = 'SM'
        wave['location'] = '01'

    return wave


def earthworm_pick_listener(debug=False):
    """
    監看 pick ring 的訊息，並保存活著的 pick msg
    pick msg 的生命週期為 p 波後 2-9 秒
    ref: pick_ew_new/pick_ra_0709.c line 283
    """
    while True:
        pick_msg = earthworm.get_msg(buf_ring=2, msg_type=0)
        if pick_msg:
            # print(pick_msg)
            try:
                pick_data = parse_pick_msg(pick_msg)
                pick_id = join_id_from_dict(pick_data, order='NSLC')
                if pick_data['update_sec'] == '2':
                    picks[pick_id] = pick_data
                    socketio.emit('pick_data', pick_data)
                    if debug:
                        print('add pick:', pick_id)
                elif pick_data['update_sec'] == '9':
                    picks.__delitem__(pick_id)
                    if debug:
                        print('delete pick:', pick_id)

            except Exception as e:
                print(e)
                continue

        time.sleep(0.00001)


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
        threading.Thread(target=trigger_emitter, daemon=True).start()
        threading.Thread(target=earthworm_convert_to_ttsam, daemon=True).start()

        if args.web or args.host or args.port:
            app.run(host=args.host, port=args.port)
            socketio.run(app, debug=True)

    except KeyboardInterrupt:
        earthworm.goodbye()
        print("Exiting...")
