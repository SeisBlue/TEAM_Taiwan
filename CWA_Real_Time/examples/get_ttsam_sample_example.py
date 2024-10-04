import json

output = {
    "waveform": [[[0.1, 0.2, 0.3]] * 3000] * 25,
    "sta": [[23.5, 121.0, 100, 760]] * 25,
    "target": [[23.5, 121.0, 100, 760]] * 25,
    "station_name": ["STA001", "STA002", "STA003", "STA004", "STA005", "STA006", "STA007", "STA008", "STA009", "STA010",
                     "STA011", "STA012", "STA013", "STA014", "STA015", "STA016", "STA017", "STA018", "STA019", "STA020",
                     "STA021", "STA022", "STA023", "STA024", "STA025"]
}

with open('../tests/data/ttsam_sample.json', 'w') as json_file:
    json.dump(output, json_file)