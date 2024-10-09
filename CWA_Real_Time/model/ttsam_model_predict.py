import json
import os

import numpy as np
import torch

from ttsam_model import get_full_model


def get_sample(json_path):
    with open(json_path, "r") as json_file:
        data = json.load(json_file)

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
    }
    return sample


if __name__ == "__main__":
    model_path = f"model/ttsam_trained_model_11.pt"
    full_Model = get_full_model(model_path)

    sample = get_sample("tests/data/ttsam_convert_2024-04-02T23:58:02.json")
    weight, sigma, mu = full_Model(sample)

    print(f"weight: {weight}")
    print(f"sigma: {sigma}")
    print(f"mu: {mu}")
