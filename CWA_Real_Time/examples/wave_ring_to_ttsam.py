import json

def wave_ring_to_ttsam(wave_ring_json):
    # Extract data from input JSON
    data = wave_ring_json["data"]

    # Transform data into the required format
    waveform = [[data[i:i+10] for i in range(0, len(data), 10)]]
    sta = [[0.0, 0.0, 0.0, 0.0]]  # [latitude, longitude, elevation (m), Vs30]
    target = [[0.0, 0.0, 0.0, 0.0]]  # [latitude, longitude, elevation (m), Vs30]
    station_name = [wave_ring_json["station"]]

    # Create the output JSON structure
    output_data = {
        "waveform": waveform,
        "sta": sta,
        "target": target,
        "station_name": station_name
    }
    return output_data


if __name__ == "__main__":
    # Read the input JSON file
    with open("tests/data/WaveRing.json", 'r') as file:
        wave_ring_json = json.load(file)
    print(wave_ring_json)

    # Generate the output JSON file
    output_data = wave_ring_to_ttsam(wave_ring_json)
    print(output_data)


