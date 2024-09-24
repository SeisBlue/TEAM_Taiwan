import PyEW
import time

"""
default ring: def_ring, 
module id: mod_id, 
installation id: inst_id, 
heartbeat interval: hb_time,
and debugging set to FALSE
"""
wave_ring = PyEW.EWModule(1000, 2, 255, 30, True)
wave_ring.add_ring(1000)
count = 0
while True:
    count += 1
    if wave_ring.mod_sta() is False:
        break
    if count > 2:
        time.sleep(0.001)
        count = 0
    wave = wave_ring.get_wave(0)
    print(wave)
