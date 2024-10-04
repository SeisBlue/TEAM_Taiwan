import bisect
import collections
import functools
import glob
import multiprocessing as mp
import os

import numpy as np
import obspy
import pandas as pd
from lxml import etree
from obspy.clients.filesystem import sds


class TaiwanIntensity:
    label = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
    pga = np.log10([1e-5, 0.008, 0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0])
    pgv = np.log10([1e-5, 0.002, 0.007, 0.019, 0.057, 0.15, 0.3, 0.5, 0.8, 1.4])

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

def read_sds(
        database_root, station, starttime, endtime, trim=True, channel="*", fmtstr=None
):
    client = sds.Client(sds_root=database_root)
    if fmtstr:
        client.FMTSTR = fmtstr
    stream = client.get_waveforms(
        network="*",
        station=station,
        location="*",
        channel=channel,
        starttime=starttime,
        endtime=endtime,
    )
    if stream:
        if trim:
            stream.trim(
                starttime, endtime, pad=True, fill_value=int(np.average(stream[0].stream))
            )
    stream.sort(keys=["channel"], reverse=True)

    stream_dict = collections.defaultdict(obspy.Stream)
    for trace in stream:
        geophone_type = trace.stats.channel[0:2]
        stream_dict[geophone_type].append(trace)
    return stream_dict


def read_gdms(
        database_root, station="*", starttime=None, endtime=None, trim=True, channel="*"
):
    fmtstr = os.path.join(
        "{year}",
        "{doy:03d}",
        "{station}",
        "{station}.{network}.{location}.{channel}.{year}.{doy:03d}",
    )

    stream_dict = read_sds(
        database_root, station, starttime, endtime, trim, channel, fmtstr
    )
    return stream_dict


def read_tsmip(txt):
    data = pd.read_fwf(txt, delim_whitespace=True, skiprows=11).to_numpy()

    with open(txt, "r") as f:
        header = f.readlines()[:11]

    stream = obspy.core.stream.Stream()

    channel = ["HLZ", "HLN", "HLE"]
    for i, chan in enumerate(channel):
        trace = obspy.core.trace.Trace(data[:, i + 1])

        trace.stats.network = "TW"
        trace.stats.station = header[0][14:20]
        trace.stats.location = "10"
        trace.stats.channel = chan

        trace.stats.starttime = obspy.UTCDateTime(header[2][12:-1])
        trace.stats.sampling_rate = int(header[4][17:20])

        stream.append(trace)

    return stream


def get_cwb_station_code(station):
    station_code = {
        "TAP": "A",
        "TCU": "B",
        "CHY": "C",
        "KAU": "D",
        "ILA": "E",
        "HWA": "F",
        "TTN": "G",
        "KNM": "I",
        "MSU": "J",
    }
    station = station_code[station[0:3]] + station[3:6]

    return station


def read_nsta(nsta24):
    with open(nsta24, "r") as file:
        seen_station = set()
        station_list = []
        lines = file.readlines()
        for line in lines:
            line = line.rstrip()

            sta = line[0:4].strip()
            lon = float(line[5:13].strip())
            lat = float(line[14:21].strip())
            elev = float(line[22:29].strip())
            loc = int(line[32:33].strip())
            source = line[34:38].strip()
            net = line[39:44].strip()
            equipments = line[45:48].strip()

            station = obspy.core.inventory.station.Station(
                code=sta,
                latitude=obspy.core.inventory.util.Latitude(lat),
                longitude=obspy.core.inventory.util.Longitude(lon),
                elevation=elev,
            )
            if sta not in seen_station and elev >= 0:
                station_list.append(station)
                seen_station.add(sta)
    return station_list


def read_header(header):
    if int(header[1:2]) == 9:
        header = header.replace("9", "199", 1)
    header_info = {
        "year": int(header[1:5]),
        "month": int(header[5:7]),
        "day": int(header[7:9]),
        "hour": int(header[9:11]),
        "minute": int(header[11:13]),
        "second": float(header[13:19]),
        "lat": float(header[19:21]),
        "lat_minute": float(header[21:26]),
        "lon": int(header[26:29]),
        "lon_minute": float(header[29:34]),
        "depth": float(header[34:40]),
        "magnitude": float(header[40:44]),
        "nsta": header[44:46].replace(" ", ""),
        "Pfilename": header[46:58].replace(" ", ""),
        "newNoPick": header[60:63].replace(" ", ""),
    }
    return header_info


def read_lines(lines):
    trace = []
    for line in lines:
        line = line.strip("\n")
        if len(line) < 109:  # missing ctime
            line = line + "   0.000"
        try:
            line_info = {
                "code": str(line[1:7]).replace(" ", ""),
                "epdis": float(line[7:13]),
                "az": int(line[13:17]),
                "phase": str(line[21:22]).replace(" ", ""),
                "ptime": float(line[23:30]),
                "pwt": int(line[30:32]),
                "stime": float(line[33:40]),
                "swt": int(line[40:42]),
                "lat": float(line[42:49]),
                "lon": float(line[49:57]),
                "gain": float(line[57:62]),
                "convm": str(line[62:63]).replace(" ", ""),
                "accf": str(line[63:75]).replace(" ", ""),
                "durt": float(line[75:79]),
                "cherr": int(line[80:83]),
                "timel": str(line[83:84]).replace(" ", ""),
                "rtcard": str(line[84:101]).replace(" ", ""),
                "ctime": str(line[101:109]).replace(" ", ""),
            }
        except ValueError:
            print(line)
            continue
        trace.append(line_info)

    return trace


def read_afile(afile):
    with open(afile) as f:
        header = f.readline()
        lines = f.readlines()
    header_info = read_header(header)
    trace_info = read_lines(lines)
    event = obspy.core.event.Event()
    event.event_descriptions.append(obspy.core.event.EventDescription())
    origin = obspy.core.event.Origin(
        time=obspy.UTCDateTime(
            header_info["year"],
            header_info["month"],
            header_info["day"],
            header_info["hour"],
            header_info["minute"],
            header_info["second"],
        ),
        latitude=header_info["lat"] + header_info["lat_minute"] / 60,
        longitude=header_info["lon"] + header_info["lon_minute"] / 60,
        depth=header_info["depth"],
    )
    origin.header = header_info
    event.origins.append(origin)

    for trace in trace_info:
        try:
            rtcard = obspy.core.UTCDateTime(trace["rtcard"])
        except Exception as err:
            print(err)
            continue

        waveform_id = obspy.core.event.WaveformStreamID(station_code=trace["code"])
        for phase in ["P", "S"]:
            if float(trace[f"{phase.lower()}time"]) == 0:
                continue

            pick = obspy.core.event.origin.Pick(
                waveform_id=waveform_id,
                phase_hint=phase,
                time=rtcard + trace[f"{phase.lower()}time"],
            )
            pick.header = trace
            event.picks.append(pick)

    event.magnitudes = header_info["magnitude"]
    return event


def isascii(s):
    """Check if the characters in string s are in ASCII, U+0-U+7F."""
    return len(s) == len(s.encode())


def read_kml_placemark(kml):
    parser = etree.XMLParser()
    root = etree.parse(kml, parser).getroot()
    geom = {}
    for Placemark in root.findall(".//Placemark", root.nsmap):
        sta = Placemark.find(".//description", root.nsmap).text[:6]
        if not isascii(sta):
            continue
        coord = Placemark.find(".//coordinates", root.nsmap).text
        coord = coord.split(",")
        location = {
            "latitude": float(coord[1]),
            "longitude": float(coord[0]),
            "elevation": float(coord[2]),
        }
        geom[sta] = location
    return geom


def get_dir_list(file_dir, suffix="", recursive=True):
    """
    Returns directory list from the given path.
    :param str file_dir: Target directory.
    :param str suffix: (Optional.) File extension, Ex: '.tfrecord'.
    :param bool recursive: (Optional.) Search directory recursively. Default is True.
    :rtype: list
    :return: List of file name.
    """
    file = os.path.join(file_dir, f"**/*{suffix}")
    file_list = glob.glob(file, recursive=recursive)
    file_list = sorted(file_list)

    return file_list


def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def batch(iterable, size=1):
    """
    Yields a batch from a list.

    :param iterable: Data list.
    :param int size: Batch size.
    """
    iter_len = len(iterable)
    for ndx in range(0, iter_len, size):
        yield iterable[ndx: min(ndx + size, iter_len)]


def batch_operation(data_list, func, **kwargs):
    """
    Unpacks and repacks a batch.

    :param data_list: List of data.
    :param func: Targeted function.
    :param kwargs: Fixed function parameter.

    :return: List of results.
    """
    return [func(data, **kwargs) for data in data_list]


def _parallel_process(file_list, par, batch_size=None, cpu_count=None):
    """
    Parallelize a partial function and return results in a list.

    :param list file_list: Process list for partial function.
    :param par: Partial function.
    :rtype: list

    :return: List of results.
    """
    if cpu_count is None:
        cpu_count = mp.cpu_count()

    pool = mp.Pool(processes=cpu_count, maxtasksperchild=1)

    if not batch_size:
        batch_size = int(np.ceil(len(file_list) / cpu_count))
    map_func = pool.imap_unordered(par, batch(file_list, batch_size))
    result = [output for output in map_func]

    pool.close()
    pool.join()
    return result


def parallel(data_list, func, batch_size=None, cpu_count=None, **kwargs):
    """
    Parallels a function.

    :param data_list: List of data.
    :param func: Paralleled function.
    :param batch_size:
    :param cpu_count:
    :param kwargs: Fixed function parameters.

    :return: List of results.
    """
    par = functools.partial(batch_operation, func=func, **kwargs)

    result_list = _parallel_process(data_list, par, batch_size, cpu_count)
    return result_list


def extract_pick_data_from_gdms(
        pick, gdms_root="/mnt/CWB", align_time=None, metadata=None
):
    # remove time shifted trace
    if pick.header.timel != "!":
        return
    try:
        accf = pick.header["accf"]
        year = pick.time.year
        month = pick.time.month

        stream = read_gdms(gdms_root)
        stream.resample(100)

        pga_thresholds = metadata["pga_thresholds"]
        pga, pga_times = get_peak_value(stream, thresholds=pga_thresholds)

        vel_stream = get_integrated_stream(stream)
        pgv_thresholds = metadata["pgv_thresholds"]
        pgv, pgv_times = get_peak_value(vel_stream, thresholds=pgv_thresholds)

        taiwan_intensity = TaiwanIntensity()
        intensity = taiwan_intensity.calculate(pga, pgv, label=True).encode("ascii")

        if pga < taiwan_intensity.pga[1]:
            return

        anchor_time = align_time
        if align_time is None:
            anchor_time = pick.time
        trim_traces(stream, anchor_time, before_len=5, after_len=25)

        waveform = get_waveform_from_stream(stream)
        station = pick.header["code"].encode("ascii")
        coords = [pick.header["lat"], pick.header["lon"], 0]
        p_picks = np.floor(((pick.time - stream[0].stats.starttime) * 100))

        pick_data_dict = {
            "station": station,
            "coords": coords,
            "waveforms": waveform,
            "p_picks": p_picks,
            "pga": pga,
            "pga_times": pga_times,
            "pgv": pgv,
            "pgv_times": pgv_times,
            "intensity": intensity,
        }
        return pick_data_dict

    except Exception as err:
        print(err)
        return


def extract_pick_data_from_tsmip(
        pick, tsmip_root="/mnt/CWB", align_time=None, metadata=None
):
    # remove time shifted trace
    if pick.header.timel != "!":
        return
    try:
        accf = pick.header["accf"]
        year = pick.time.year
        month = pick.time.month

        stream = read_tsmip(f"{tsmip_root}/{year}/{month:0>2}/{accf}.txt")
        stream.resample(100)

        pga_thresholds = metadata["pga_thresholds"]
        pga, pga_times = get_peak_value(stream, thresholds=pga_thresholds)

        vel_stream = get_integrated_stream(stream)
        pgv_thresholds = metadata["pgv_thresholds"]
        pgv, pgv_times = get_peak_value(vel_stream, thresholds=pgv_thresholds)

        taiwan_intensity = TaiwanIntensity()
        intensity = taiwan_intensity.calculate(pga, pgv, label=True).encode("ascii")

        if pga < taiwan_intensity.pga[1]:
            return

        anchor_time = align_time
        if align_time is None:
            anchor_time = pick.time
        trim_traces(stream, anchor_time, before_len=5, after_len=25)

        waveform = get_waveform_from_stream(stream)
        station = pick.header["code"].encode("ascii")
        coords = [pick.header["lat"], pick.header["lon"], 0]
        p_picks = np.floor(((pick.time - stream[0].stats.starttime) * 100))

        pick_data_dict = {
            "station": station,
            "coords": coords,
            "waveforms": waveform,
            "p_picks": p_picks,
            "pga": pga,
            "pga_times": pga_times,
            "pgv": pgv,
            "pgv_times": pgv_times,
            "intensity": intensity,
        }
        return pick_data_dict

    except Exception as err:
        print(err)
        return


def get_waveform_from_stream(stream):
    waveform = np.zeros((3000, 3))
    min_len = min(3000, len(stream.traces[0].stream))

    stream.sort(["component"])
    for i, component in enumerate(["N", "E", "Z"]):
        waveform[:min_len, i] = stream.select(component=component)[0].stream[:min_len]

    return waveform


def trim_traces(stream, anchor_time=None, before_len=5, after_len=25):
    stream.trim(
        starttime=anchor_time - before_len,
        endtime=anchor_time + after_len,
        pad=True,
        fill_value=0,
    )


def get_peak_value(stream, thresholds=None):
    data = [tr.stream for tr in stream]
    data = np.array(data)
    vector = np.linalg.norm(data, axis=0)

    peak = max(vector)
    peak = np.log10(peak / 100)

    exceed_times = np.zeros(5)
    if thresholds is not None:
        for i, threshold in enumerate(thresholds):
            try:
                exceed_times[i] = next(
                    x for x, val in enumerate(vector) if val > threshold
                )
            except Exception as err:
                print(err)

    return peak, exceed_times


def get_integrated_stream(stream):
    stream_vel = stream.copy()
    stream_vel.filter("bandpass", freqmin=0.075, freqmax=10)
    stream_vel.integrate()
    return stream_vel
